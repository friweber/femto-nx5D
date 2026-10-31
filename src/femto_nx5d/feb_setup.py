"""
Config und Notebooks für die FemtoMAX-Strahlzeit Feb 2025 (Schichtproben, ein Peak pro Schicht).

    femto-nx5d setup-feb --raw <rohdaten-ordner> --out <arbeitsordner> [--scan-files <ordner>]

- Proben-Sets: standardmäßig die korrigierten Scanlisten der 8 Paper-Proben (400 nm) aus dem Paket
  (femto_nx5d/data/feb2025_scan_lists.json: Scans an falscher Probenposition umsortiert, zusätzliche
  vollständige Scans als eigene Gruppe). Eigene Listen `measurement_file_<Set>.txt` (Spalten: Scan,
  Probe, Scantyp, Frames, Wellenlänge, Winkel, Leistung) haben Vorrang: --scan-files oder
  <out>/scan_files; --search-share sucht sie unter dem Strahlzeit-Ordner (Eval*/…/scan_files).
  Scantyp -> Gruppe: 0 = Au-Winkel, 1 = Ni-Winkel, 3 = Cu-Winkel, 2 = Scans einer anderen Probe
  (ohne Fit), 10/11/13 = zusätzliche Scans am Au-/Ni-/Cu-Winkel (Gruppe "<X>_zusatz", getrennt gefittet).
- RSM-Scans (e_gon01_keta) pro Probe über which_sample.csv neben dem Rohdaten-Ordner.
- Geometrie wie in der bisherigen Profil-Auswertung (2_1_preprocess_profiles): Detektor um 90°
  gedreht, a_in = |keta + 30.7|, a_out = Austrittswinkel + atan((x - Mittelpixel) 0.172/450)
  (x = Pixel des gedrehten Bildes), λ = 1.5498 Å, Ping-t0 45797.86 ps bzw. ab Scan 125549 45803.0 ps.
  Mittelpixel und Austrittswinkel wurden damals pro Scan eingestellt (z. B. 670/18.5° bei keta -50°,
  950/21.25° bei -53.5°); sie werden aus den vorhandenen process/scan-N_roi_profile.nc gelesen
  (--process, sonst <Strahlzeit>/process), damit die q-Skala gleich bleibt. Ohne diese Dateien:
  Detektor fest im Labor, 2θ = 37.8° bei Pixel 670.
Bestehende Dateien werden nicht überschrieben (außer mit --force).
"""
import argparse
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np

SCANTYPE_GROUP = {0: "Au", 1: "Ni", 2: "Typ2", 3: "Cu", 10: "Au_zusatz", 11: "Ni_zusatz", 13: "Cu_zusatz"}
# q_z (1/Å) on the scale of the earlier evaluation (λ = 1.5498 Å) and fit half width
PEAKS = {"Au": (2.64, 0.07), "Ni": (3.04, 0.06), "Cu": (2.97, 0.07)}


def builtin_scan_lists():
    """Corrected scan lists shipped with the package -> ({set: array like read_scan_files}, {set: notes})."""
    from importlib import resources
    data = json.loads(resources.files("femto_nx5d").joinpath("data/feb2025_scan_lists.json").read_text(encoding="utf-8"))
    sets, notes = {}, {}
    for name, st in data["sets"].items():
        a = np.full((len(st["rows"]), 7), np.nan)
        for i, (scan, t, ang) in enumerate(st["rows"]):
            a[i, 0], a[i, 2], a[i, 4], a[i, 5] = scan, t, data.get("pump_wavelength_nm", np.nan), ang
        sets[name], notes[name] = a, st.get("notes", [])
    return sets, notes


def builtin_geometry():
    """Per-scan [center_pixel, exit_angle] of the earlier evaluation, shipped with the package."""
    from importlib import resources
    data = json.loads(resources.files("femto_nx5d").joinpath("data/feb2025_scan_lists.json").read_text(encoding="utf-8"))
    return data.get("scan_geometry", {})


def find_scan_files(raw, out, given=None, max_depth=5, search=False):
    cands = [Path(given)] if given else []
    cands.append(Path(out) / "scan_files")
    for c in cands:
        if c.is_dir() and list(c.glob("measurement_file_*.txt")):
            return c
    if not search:
        return None
    base = Path(raw).resolve().parent.parent          # .../<proposal>/<visit>
    found = []
    for root, dirs, files in os.walk(base):
        depth = len(Path(root).relative_to(base).parts)
        if depth == 0:
            dirs[:] = [d for d in dirs if d.lower().startswith(("eval", "process", "analysis"))]
        if depth >= max_depth or "raw" in Path(root).parts[len(base.parts):]:
            dirs[:] = []
        if any(f.startswith("measurement_file_") and f.endswith(".txt") for f in files):
            found.append(Path(root))
    if not found:
        return None
    # the folder with the most scan lists (usually Eval_by_Project/<…>/scan_files)
    return max(found, key=lambda d: (len(list(d.glob("measurement_file_*.txt"))), -len(d.parts)))


def read_scan_files(d):
    sets = {}
    for f in sorted(Path(d).glob("measurement_file_*.txt")):
        a = np.genfromtxt(f, ndmin=2, comments="#")
        if a.size == 0:
            continue
        sets[f.stem[len("measurement_file_"):]] = a
    return sets


def _which_sample(raw):
    p = Path(raw).resolve().parent / "which_sample.csv"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8-sig").splitlines()[1:]:
        parts = [x.strip() for x in line.split(",")]
        try:
            rows.append((int(float(parts[0])), int(float(parts[1])), parts[2]))
        except (ValueError, IndexError):
            pass
    return rows


def rsm_scans(raw, set_name, ranges, max_scans=3):
    """keta scans in the which_sample ranges of the sample (e.g. NiAu100_400 -> 'NiAu(100)')."""
    base = set_name.rsplit("_", 1)[0]
    label = re.sub(r"^(\D+?)(\d+)$", r"\1(\2)", base)
    found = []
    for a, b, name in ranges:
        if name != label:
            continue
        for s in range(a, b + 1):
            m = Path(raw) / f"scan-{s}.h5"
            if not m.exists():
                continue
            try:
                with h5py.File(m, "r") as f:
                    mot = f["scan/motor1_name"][()]
                    mot = mot.decode() if isinstance(mot, bytes) else str(mot)
                    n = len(f[f"entry/measurement/{mot}"]) if f"entry/measurement/{mot}" in f else 0
            except Exception:
                continue
            if mot == "e_gon01_keta" and n > 5:
                found.append((n, s))
    return [s for n, s in sorted(found, key=lambda x: (-x[0], x[1]))[:max_scans]]


PX, DIST = 0.172, 450.0           # pixel size, detector distance (mm)
TWO_THETA_LAB = 37.8               # 2theta at pixel 670 of the rotated image (a_in 19.3 + 18.5)


def find_process(raw, given=None):
    for c in ([Path(given)] if given else []) + [Path(raw).resolve().parent.parent / "process"]:
        if c.is_dir() and next(c.glob("scan-*_roi_profile.nc"), None) is not None:
            return c
    return None


def old_geometry(process, scans):
    """{scan: [center_pixel, exit_angle]} fitted to x/theta of the earlier roi_profile files."""
    from scipy.optimize import curve_fit
    f = lambda x, c, det: np.degrees(np.arctan((x - c) * PX / DIST)) + det
    out = {}
    for s in scans:
        p = Path(process) / f"scan-{int(s)}_roi_profile.nc"
        if not p.exists():
            continue
        try:
            with h5py.File(p, "r") as h:
                x, th = h["x"][()].astype(float), h["theta"][()].astype(float)
            (c, det), _ = curve_fit(f, x, th, p0=(700.0, 19.0))
            out[str(int(s))] = [round(float(c), 2), round(float(det), 4)]
        except Exception:
            continue
    return out


def build_config(raw, out, sets, process=None, notes=None, paper=None):
    raw, out = Path(raw).resolve(), Path(out).resolve()
    ranges = _which_sample(raw)
    all_scans = sorted({int(s) for a in sets.values() for s in a[:, 0]})
    proc = find_process(raw, process)
    scan_geo = {k: v for k, v in builtin_geometry().items() if int(k) in all_scans}
    if proc:
        scan_geo.update(old_geometry(proc, all_scans))
    # plausibility: 2theta at pixel 800 (a_in + exit angle + pixel angle) within 1 deg of the lab geometry
    keta_of = {int(r[0]): float(r[5]) for a in sets.values() for r in a}
    tth_lab = TWO_THETA_LAB + np.degrees(np.arctan((800 - 670) * PX / DIST))
    rejected = []
    for k in list(scan_geo):
        c, det = scan_geo[k]
        tth = (-keta_of.get(int(k), np.nan) - 30.7) + det + np.degrees(np.arctan((800 - c) * PX / DIST))
        if not abs(tth - tth_lab) <= 1.0:
            rejected.append(int(k))
            del scan_geo[k]
    cfg_sets = {}
    for name, a in sets.items():
        groups = {}
        for t in sorted({int(x) for x in a[:, 2]}):
            groups[SCANTYPE_GROUP.get(t, f"Typ{t}")] = [int(s) for s in a[a[:, 2] == t, 0]]
        layer = "Ni" if name.startswith("Ni") else "Cu" if name.startswith("Cu") else None
        elements = [e for e in ("Au", layer) if e]
        peaks = {}
        for g in groups:
            base = g.split("_")[0]
            if base not in PEAKS:
                continue                       # e.g. Typ2: no default fit
            for e in [base] + [x for x in elements if x != base]:
                q, hw = PEAKS[e]
                peaks[e if e == g else f"{e}@{g}"] = {"element": e, "group": g, "qz": q, "halfwidth": hw}
        wl = name.rsplit("_", 1)[-1]
        geometry = {}
        for g in sorted(groups, key=lambda g: "_" in g):          # base groups first
            scans = groups[g]
            known = [scan_geo[str(s)] for s in scans if str(s) in scan_geo]
            if known:
                geometry[g] = [float(np.median([k[0] for k in known])), float(np.median([k[1] for k in known]))]
            elif g.split("_")[0] in geometry:                     # extra scans: like their base group
                geometry[g] = list(geometry[g.split("_")[0]])
            else:     # detector fixed in the lab, exit angle at pixel 670 = 2theta - a_in
                keta = float(np.median(a[np.isin(a[:, 0], scans), 5]))
                geometry[g] = [670.0, round(TWO_THETA_LAB - (-keta - 30.7), 4)]
        set_notes = list((notes or {}).get(name, []))
        for g, scans in groups.items():
            bad = [x for x in scans if x in rejected]
            if bad:
                set_notes.append(f"{g}: alte Geometrie von {bad} unplausibel (2θ > 1° neben der Laborgeometrie) "
                                 f"-> Gruppenwert {geometry[g]}")
        cfg_sets[name] = {"groups": groups, "peaks": peaks, "rsm_scans": rsm_scans(raw, name, ranges),
                          "geometry": geometry, "delay_offset": 0.0,
                          "paper": (wl == "400") if paper is None else paper,
                          "pump_wavelength_nm": wl, "notes": set_notes}
    prop = raw.name                             # e.g. "NiAu" (collection name)
    return {
        "_comment": "FemtoMAX Feb 2025 (femto_nx5d.feb_setup). Geometrie wie in 2_1_preprocess_profiles. "
                    "Pro Set: groups = Scans je Winkel (Scantyp), peaks = Fits (Label 'X@Gruppe' = Peak X in "
                    "den Scans der Gruppe), delay_offset = t0-Korrektur (ps, früher t0_offset).",
        "raw_dirs": [str(raw)],
        "proposal_root": str(out / "proposals"),
        "proposal": prop,
        "catalog_file": "catalog.csv",
        "results_dir": "results",
        "roi": {"rows": [100, 701], "cols": [474, 975]},
        "_roi_comment": "Pilatus-Rohpixel; entspricht x 500-1000 / y 100-700 des gedrehten Bildes der alten Auswertung",
        "scatter_axis": "cols",
        "mask": {"dilate_gaps": True, "max_count": None},
        "read_mode": "roi",
        "ping_channel": "Ping_Ch2_value",
        "ping_t0_ps": [[0, 45797.86], [125549, 45803.0]],
        "theta_motor": "e_gon01_keta",
        "theta_sign": -1,
        "geometry": "fixed_exit",
        "loop_mode": "merge",
        "scan_types": {"xraydelay": "tRSS", "e_gon01_keta": "sRSM", "l_lwp01_energy": "fRSS"},
        "samples": {prop: {"e_gon01_x": 0.0, "beamEnergy": 8000.0, "center": {"row": 300, "col": 330},
                           "imageDistance": DIST, "theta_offset": -30.7, "ttheta_offset": 18.5}},
        "_samples_comment": "Grundgeometrie aller Scans (ein nx5d-Proposal): a_in = -keta - 30.7, a_out = 18.5° + "
                            "Pixelwinkel, Mitte = ROI-Spalte 330 (= Pixel 670 des gedrehten Bildes), "
                            "λ = 12398.42/8000 Å = 1.5498 Å. Pro Scan überschrieben durch scan_geometry bzw. "
                            "sets[...].geometry (use_scan_geometry).",
        "two_theta_lab": TWO_THETA_LAB,
        "use_scan_geometry": True,
        "old_pixel_to_roi": 1000,
        "_scan_geometry_comment": "[Mittelpixel, Austrittswinkel] je Scan wie in 2_1_preprocess_profiles "
                                  "(Pixel des gedrehten Bildes); aus dem Paket" + (f" und {proc}" if proc else ""),
        "scan_geometry": scan_geo,
        "delay_bins_ps": [[-5, 50, 0.5], [50, 200, 1.0]],
        "strain_ref_ps": [-5, 0],
        "qz_halfwidth": 0.07,
        "sets": cfg_sets,
    }


def write(out, cfg, force=False):
    from . import notebooks
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    path = out / "femtomax_config.json"
    if force or not path.exists():
        path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
        written.append(path)
    for name in cfg["sets"]:
        p = out / f"{name}_analysis.ipynb"
        if force or not p.exists():
            written.append(notebooks.make_feb(name, out))
    p = out / "Alle_Proben.ipynb"
    if force or not p.exists():
        written.append(notebooks.make_feb_overview(out))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(prog="femto-nx5d setup-feb", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="Rohdaten-Ordner (scan-N.h5 …)")
    ap.add_argument("--out", default=".", help="Arbeitsordner (Config, Notebooks, Ergebnisse)")
    ap.add_argument("--scan-files", help="Ordner mit measurement_file_<Set>.txt")
    ap.add_argument("--process", help="Ordner mit den bisherigen scan-N_roi_profile.nc (Geometrie je Scan)")
    ap.add_argument("--search-share", action="store_true",
                    help="eigene Scanlisten unter dem Strahlzeit-Ordner (Eval*/…/scan_files) suchen")
    ap.add_argument("--force", action="store_true", help="vorhandene Config/Notebooks überschreiben")
    a = ap.parse_args(argv)
    d = find_scan_files(a.raw, a.out, a.scan_files, search=a.search_share)
    if d is None:
        sets, notes = builtin_scan_lists()
        print(f"Scanlisten: korrigierte Paper-Listen aus dem Paket ({len(sets)} Sets, 400 nm)")
        for n, ns in notes.items():
            for x in ns:
                print(f"  {n}: {x}")
    else:
        sets, notes = read_scan_files(d), {}
        print(f"Scanlisten: {d} ({len(sets)} Sets)")
    cfg = build_config(a.raw, a.out, sets, a.process, notes)
    print(f"Geometrie je Scan wie in der bisherigen Auswertung: {len(cfg['scan_geometry'])} Scans"
          + ("" if cfg["scan_geometry"] else " (keine gefunden -> Labor-Geometrie)"))
    for n, st in cfg["sets"].items():
        print(f"  {n:16s} Gruppen {', '.join(f'{g} ({len(v)})' for g, v in st['groups'].items())}; "
              f"Peaks {', '.join(st['peaks'])}; RSM {st['rsm_scans'] or '-'}{'; Paper' if st['paper'] else ''}; "
              f"Geometrie {st['geometry']}")
    for p in write(a.out, cfg, a.force):
        print("geschrieben:", p)
    print("Weiter: Alle_Proben.ipynb (alle Paper-Proben) oder <Set>_analysis.ipynb öffnen und ausführen.")


if __name__ == "__main__":
    main()
