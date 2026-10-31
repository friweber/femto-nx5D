"""
Config und Notebooks für die FemtoMAX-Strahlzeit Feb 2025 (Schichtproben, ein Peak pro Schicht).

    femto-nx5d setup-feb --raw <rohdaten-ordner> --out <arbeitsordner> [--scan-files <ordner>]

- Proben-Sets kommen aus den Scanlisten `measurement_file_<Set>.txt` (Spalten: Scan, Probe,
  Scantyp, Frames, Wellenlänge, Winkel, Leistung). Der Scantyp legt die Gruppe fest
  (0 = Au-Winkel, 1 = Ni-Winkel, 3 = Cu-Winkel, 2 = weitere Au-Winkel-Scans, ohne Standard-Fit).
- Gesucht wird in --scan-files, <out>/scan_files und unter dem Strahlzeit-Ordner (Eval*/…/scan_files).
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

SCANTYPE_GROUP = {0: "Au", 1: "Ni", 2: "Typ2", 3: "Cu"}
# q_z (1/Å) on the scale of the earlier evaluation (λ = 1.5498 Å) and fit half width
PEAKS = {"Au": (2.64, 0.07), "Ni": (3.04, 0.06), "Cu": (2.97, 0.07)}


def find_scan_files(raw, out, given=None, max_depth=5):
    cands = [Path(given)] if given else []
    cands.append(Path(out) / "scan_files")
    for c in cands:
        if c.is_dir() and list(c.glob("measurement_file_*.txt")):
            return c
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


def build_config(raw, out, sets, process=None):
    raw, out = Path(raw).resolve(), Path(out).resolve()
    ranges = _which_sample(raw)
    all_scans = sorted({int(s) for a in sets.values() for s in a[:, 0]})
    proc = find_process(raw, process)
    scan_geo = old_geometry(proc, all_scans) if proc else {}
    cfg_sets = {}
    for name, a in sets.items():
        groups = {}
        for t in sorted({int(x) for x in a[:, 2]}):
            groups[SCANTYPE_GROUP.get(t, f"Typ{t}")] = [int(s) for s in a[a[:, 2] == t, 0]]
        layer = "Ni" if name.startswith("Ni") else "Cu" if name.startswith("Cu") else None
        elements = [e for e in ("Au", layer) if e]
        peaks = {}
        for g in groups:
            if g not in PEAKS:
                continue                       # e.g. Typ2: no default fit
            for e in [g] + [x for x in elements if x != g]:
                q, hw = PEAKS[e]
                peaks[e if e == g else f"{e}@{g}"] = {"element": e, "group": g, "qz": q, "halfwidth": hw}
        wl = name.rsplit("_", 1)[-1]
        geometry = {}
        for g, scans in groups.items():
            known = [scan_geo[str(s)] for s in scans if str(s) in scan_geo]
            if known:
                geometry[g] = [float(np.median([k[0] for k in known])), float(np.median([k[1] for k in known]))]
            else:     # detector fixed in the lab, exit angle at pixel 670 = 2theta - a_in
                keta = float(np.median(a[np.isin(a[:, 0], scans), 5]))
                geometry[g] = [670.0, round(TWO_THETA_LAB - (-keta - 30.7), 4)]
        cfg_sets[name] = {"groups": groups, "peaks": peaks, "rsm_scans": rsm_scans(raw, name, ranges),
                          "geometry": geometry, "delay_offset": 0.0, "paper": wl == "400",
                          "pump_wavelength_nm": wl}
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
                                  "(Pixel des gedrehten Bildes); aus " + (str(proc) if proc else "- (keine process-Dateien)"),
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
    ap.add_argument("--force", action="store_true", help="vorhandene Config/Notebooks überschreiben")
    a = ap.parse_args(argv)
    d = find_scan_files(a.raw, a.out, a.scan_files)
    if d is None:
        raise SystemExit("Keine Scanlisten (measurement_file_<Set>.txt) gefunden. Den Ordner scan_files "
                         f"nach {Path(a.out).resolve() / 'scan_files'} kopieren oder --scan-files angeben.")
    sets = read_scan_files(d)
    print(f"Scanlisten: {d} ({len(sets)} Sets)")
    cfg = build_config(a.raw, a.out, sets, a.process)
    print(f"Geometrie je Scan aus bisherigen Profilen: {len(cfg['scan_geometry'])} Scans"
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
