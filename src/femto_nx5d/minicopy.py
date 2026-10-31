"""
Verkleinerte Kopie einzelner Scans zum Testen (z. B. auf einem anderen Rechner):
Master-Datei unverändert, vom Pilatus-Stapel nur die ersten `--frames` Bilder und darin nur
die ROI (Rest 0, gleiche Bildgröße), von der Oszilloskop-Datei nur die Werte pro Bild
(keine Spuren). Zyla-Dateien werden nicht kopiert.

    python -m femto_nx5d.minicopy <raw-ordner> <ziel-ordner> SCAN [SCAN ...]
           [--frames 200] [--rows 100 701] [--cols 474 975]

Schreibt <ziel-ordner>/mini/scan-N*.h5 und <ziel-ordner>/mini_scans.tar.
"""
import argparse
import os
import shutil
import tarfile

import h5py
import numpy as np

PIL = "entry/instrument/pilatus"


def _copy_small(src, dst, max_size=10**6):
    """Kopiert Gruppen/Datensätze rekursiv, große Datensätze werden ausgelassen."""
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    for k in src:
        o = src.get(k)
        if isinstance(o, h5py.Group):
            _copy_small(o, dst.require_group(k), max_size)
        elif isinstance(o, h5py.Dataset) and o.size <= max_size:
            dst.create_dataset(k, data=o[()])


def minicopy(raw, out, scan, frames=200, rows=(100, 701), cols=(474, 975), margin=2):
    os.makedirs(out, exist_ok=True)
    stem = f"scan-{int(scan)}"
    shutil.copy2(os.path.join(raw, stem + ".h5"), os.path.join(out, stem + ".h5"))
    written = [stem + ".h5"]

    src = os.path.join(raw, stem + "_pilatus.h5")
    if os.path.exists(src):
        with h5py.File(src, "r") as fi, h5py.File(os.path.join(out, stem + "_pilatus.h5"), "w") as fo:
            g = fi[PIL]
            _copy_small(g, fo.require_group(PIL))
            d = g["data"]
            n = min(int(frames), d.shape[0])
            H, W = d.shape[1:]
            r0, r1 = max(rows[0] - margin, 0), min(rows[1] + margin, H)
            c0, c1 = max(cols[0] - margin, 0), min(cols[1] + margin, W)
            o = fo[PIL].create_dataset("data", shape=(n, H, W), dtype=d.dtype, chunks=(1, H, W),
                                       compression="gzip", compression_opts=4, shuffle=True)
            frame = np.zeros((H, W), d.dtype)
            for i in range(n):
                frame[:] = 0
                frame[r0:r1, c0:c1] = d[i, r0:r1, c0:c1]
                o[i] = frame
            fo[PIL].attrs["minicopy"] = f"frames 0..{n - 1}, rows {r0}:{r1}, cols {c0}:{c1}, rest 0"
        written.append(stem + "_pilatus.h5")

    src = os.path.join(raw, stem + "_oscc_02_maui.h5")
    if os.path.exists(src):
        with h5py.File(src, "r") as fi, h5py.File(os.path.join(out, stem + "_oscc_02_maui.h5"), "w") as fo:
            _copy_small(fi, fo)
        written.append(stem + "_oscc_02_maui.h5")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("scans", nargs="+", type=int)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--rows", type=int, nargs=2, default=(100, 701))
    ap.add_argument("--cols", type=int, nargs=2, default=(474, 975))
    a = ap.parse_args()
    mini = os.path.join(a.out, "mini")
    files = []
    for s in a.scans:
        files += minicopy(a.raw, mini, s, a.frames, a.rows, a.cols)
        print(s, "ok", flush=True)
    tar = os.path.join(a.out, "mini_scans.tar")
    with tarfile.open(tar, "w") as t:
        for f in files:
            t.add(os.path.join(mini, f), arcname=f)
    print("fertig:", tar, round(os.path.getsize(tar) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
