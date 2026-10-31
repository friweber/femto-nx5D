"""
Profil-Auswertung (ROI-Profile, Delay-Scans) für FemtoMAX-Sardana-Scans ohne nx5d.

- Pfade kommen aus einer Config (JSON), standardmäßig ./feb_config.json im Ordner des
  Notebooks oder die Datei in $FEMTO_FEB_CONFIG:

      {"raw_dirs":     ["/pfad/zu/raw", "..."],          # gesucht in dieser Reihenfolge
       "process_dirs": ["process", "/pfad/zu/process"],  # vorverarbeitete .nc, eigener Ordner zuerst
       "process_out":  "process"}                         # hierhin wird geschrieben

  Relative Pfade gelten relativ zur Config-Datei. Ohne Config: ./raw und ./process.
- read_roi: liest aus dem Pilatus-Stapel nur die ROI, blockweise, statt den ganzen Stapel
  in den Speicher zu laden. Maskierung (Modullücken aus Bild 0, 3x3 dilatiert; Werte < 0;
  optional Dark-Maske) und Drehung wie in der bisherigen process_scan-Routine.

    import femto_nx5d.feb as P
    P.raw("scan-123456.h5"); P.processed("scan-123456_roi_profile.nc")
"""
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation

RAW_DIRS, PROCESS_DIRS, PROCESS_OUT, CONFIG = [], [], None, None


def configure(path=None, raw_dirs=None, process_dirs=None, process_out=None):
    """
    Setzt die Suchpfade. path: JSON-Config (Standard: $FEMTO_FEB_CONFIG oder ./feb_config.json);
    einzelne Argumente überschreiben die Config. Zusätzlich vorne angefügt:
    $FEB_RAW_DIRS / $FEB_PROCESS_DIRS (durch ":" getrennt).
    """
    global RAW_DIRS, PROCESS_DIRS, PROCESS_OUT, CONFIG
    path = path or os.environ.get("FEMTO_FEB_CONFIG") or "feb_config.json"
    path = Path(path).expanduser()
    cfg, base = {}, Path.cwd()
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        CONFIG = path.resolve()
    else:
        CONFIG = None
    absolute = lambda p: (Path(p).expanduser() if Path(p).expanduser().is_absolute()
                          else base / p)
    out = absolute(process_out or cfg.get("process_out", "process"))
    RAW_DIRS = [absolute(p) for p in (raw_dirs or cfg.get("raw_dirs", ["raw"]))]
    PROCESS_DIRS = [absolute(p) for p in (process_dirs or cfg.get("process_dirs", [str(out)]))]
    if out not in PROCESS_DIRS:
        PROCESS_DIRS.insert(0, out)
    RAW_DIRS[:0] = [Path(p) for p in os.environ.get("FEB_RAW_DIRS", "").split(":") if p]
    PROCESS_DIRS[1:1] = [Path(p) for p in os.environ.get("FEB_PROCESS_DIRS", "").split(":") if p]
    PROCESS_OUT = out
    return {"config": CONFIG, "raw_dirs": RAW_DIRS, "process_dirs": PROCESS_DIRS,
            "process_out": PROCESS_OUT}


configure()

PILATUS_PATH = "entry/instrument/pilatus/data"
MIN_AGE_S = 600   # jünger = wird evtl. noch übertragen (z. B. Globus)


def raw(name, min_age_s=MIN_AGE_S):
    """Pfad einer Rohdatei (z. B. 'scan-124958.h5'); Fehler, wenn nicht (vollständig) da."""
    for d in RAW_DIRS:
        p = d / name
        if p.exists():
            age = time.time() - p.stat().st_mtime
            if age < min_age_s:
                raise FileNotFoundError(f"{p} wurde vor {age:.0f} s geändert - Übertragung evtl. "
                                        f"noch nicht fertig")
            return str(p)
    raise FileNotFoundError(f"{name} nicht gefunden in: " + ", ".join(map(str, RAW_DIRS)))


def processed(name):
    """Pfad einer vorverarbeiteten Datei, eigener Ordner zuerst."""
    for d in PROCESS_DIRS:
        p = d / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"{name} nicht gefunden in: " + ", ".join(map(str, PROCESS_DIRS)))


def source_of(name):
    """'eigen' oder 'Beamtime' - woher eine vorverarbeitete Datei kommt."""
    return "eigen" if Path(processed(name)).parent == PROCESS_OUT else "Beamtime"


def _pilatus(scan_nr):
    """Öffnet den Pilatus-Datensatz direkt (ohne Chunk-Cache: liest nur die ROI-Bytes)."""
    try:
        f = h5py.File(raw(f"scan-{int(scan_nr)}_pilatus.h5"), "r", rdcc_nbytes=0)
        return f, f[PILATUS_PATH]
    except (FileNotFoundError, KeyError):
        f = h5py.File(raw(f"scan-{int(scan_nr)}.h5"), "r", rdcc_nbytes=0)
        return f, f["/entry/measurement/pilatus/data"]


def _mask(block, gaps):
    block = block.astype(np.float32, copy=False) if block.dtype.kind != "f" else block
    block[:, gaps] = np.nan
    block[block < 0] = np.nan
    return block


def read_roi(scan_nr, n_frames, roi_x, roi_y, rotate, dark_mask=None,
             overview_frames=0, block=32):
    """
    ROI eines Scans wie in process_scan:
        images = pilatus[:n_frames]; Lücken (Bild 0 == -1, 3x3 dilatiert), Werte < 0
        und Dark-Maske -> NaN; bei rotate np.rot90(k=1, axes=(1, 2));
        dann .sel(x=slice(*roi_x), y=slice(*roi_y)).

    Rückgabe: images (frames, x, y), x_idx, y_idx, overview
        overview = Mittel der ersten `overview_frames` ganzen Bilder (gedreht, maskiert)
        oder None.
    """
    f, d = _pilatus(scan_nr)
    try:
        N, H, W = d.shape
        n = min(int(n_frames), N)
        if rotate:   # rot90(k=1): Bild[x, y] = Original[Zeile y, Spalte W-1-x]
            x_idx = np.arange(max(roi_x[0], 0), min(roi_x[1], W - 1) + 1)
            y_idx = np.arange(max(roi_y[0], 0), min(roi_y[1], H - 1) + 1)
            rows = (y_idx[0], y_idx[-1] + 1)
            cols = (W - 1 - x_idx[-1], W - x_idx[0])
        else:        # Bild[x, y] = Original[Zeile x, Spalte y]
            x_idx = np.arange(max(roi_x[0], 0), min(roi_x[1], H - 1) + 1)
            y_idx = np.arange(max(roi_y[0], 0), min(roi_y[1], W - 1) + 1)
            rows = (x_idx[0], x_idx[-1] + 1)
            cols = (y_idx[0], y_idx[-1] + 1)

        # Lückenmaske aus Bild 0, mit 1 px Rand, damit die Dilatation am ROI-Rand stimmt
        R0, R1 = max(rows[0] - 1, 0), min(rows[1] + 1, H)
        C0, C1 = max(cols[0] - 1, 0), min(cols[1] + 1, W)
        gaps = binary_dilation(d[0, R0:R1, C0:C1] == -1, structure=np.ones((3, 3)))
        gaps = gaps[rows[0] - R0:rows[1] - R0, cols[0] - C0:cols[1] - C0]
        if dark_mask is not None:
            gaps = gaps | np.asarray(dark_mask, bool)[rows[0]:rows[1], cols[0]:cols[1]]

        out = None
        for i in range(0, n, block):
            a = _mask(d[i:min(i + block, n), rows[0]:rows[1], cols[0]:cols[1]], gaps)
            if out is None:
                out = np.empty((n,) + a.shape[1:], a.dtype)
            out[i:i + len(a)] = a
        if rotate:
            out = np.rot90(out, k=1, axes=(1, 2))

        overview = None
        m = min(int(overview_frames), n)
        if m > 0:
            g = binary_dilation(d[0] == -1, structure=np.ones((3, 3)))
            if dark_mask is not None:
                g = g | np.asarray(dark_mask, bool)
            s = np.zeros((H, W)); c = np.zeros((H, W))
            for i in range(0, m, block):
                a = _mask(d[i:min(i + block, m)], g)
                s += np.nansum(a, axis=0); c += np.sum(~np.isnan(a), axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                overview = np.where(c > 0, s / c, np.nan)
            if rotate:
                overview = np.rot90(overview, k=1)
        return out, x_idx, y_idx, overview
    finally:
        f.close()


def status(scan_numbers):
    """Tabelle: welche Scans Rohdaten / Profile haben."""
    rows = []
    for s in scan_numbers:
        def ok(fn, name):
            try:
                fn(name)
                return True
            except FileNotFoundError:
                return False
        prof = ok(processed, f"scan-{s}_roi_profile.nc")
        rows.append((s, ok(raw, f"scan-{s}.h5") and ok(raw, f"scan-{s}_pilatus.h5"),
                     prof, source_of(f"scan-{s}_roi_profile.nc") if prof else "-"))
    try:
        import pandas as pd
        return pd.DataFrame(rows, columns=["scan", "Rohdaten", "Profil", "Profil aus"]).set_index("scan")
    except ImportError:
        return rows
