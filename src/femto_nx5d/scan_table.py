"""
Kompakte Scan-Tabelle aller Sardana-Master-Dateien eines Rohdaten-Ordners: nur Skalare
(scan/*, meta/*, pre_scan_snapshot, erste/letzte Motorposition), keine Bilder; externe Dateien
(Pilatus, Ping, Zyla) werden nicht geöffnet. Dazu die HDF5-Struktur je Scan-Typ.

    python -m femto_nx5d.scan_table <raw-ordner> <ziel-ordner>

Schreibt scan_table.csv, scan_trees.txt und scan_table.tar.gz (beide + which_sample.csv /
scan_daten.txt aus dem Ordner über raw, falls vorhanden) nach <ziel-ordner>.
"""
import os
import re
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
import h5py, numpy as np, pandas as pd

RAW = OUT = None


def _v(d):
    v = d[()]
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray):
        if v.size != 1:
            return None
        v = v.item()
        return v.decode(errors="replace") if isinstance(v, bytes) else v
    return v


def _init(raw):
    global RAW
    RAW = raw


def one(name):
    r = {"scan": int(name[5:-3])}
    try:
        with h5py.File(os.path.join(RAW, name), "r") as f:
            for g in ("scan", "meta"):
                if g in f:
                    for k in f[g]:
                        d = f[g].get(k, getlink=True)
                        if not isinstance(d, h5py.ExternalLink) and isinstance(f[g].get(k), h5py.Dataset):
                            r[f"{g}/{k}"] = _v(f[g][k]) if f[g][k].size == 1 else f"array{f[g][k].shape}"
            for k in ("entry/title", "entry/start_time", "entry/end_time", "entry/entry_identifier"):
                if k in f:
                    r[k] = _v(f[k])
            snap = f.get("entry/measurement/pre_scan_snapshot")
            if snap is not None:
                for k in snap:
                    if isinstance(snap.get(k), h5py.Dataset) and snap[k].size == 1:
                        r[f"snap/{k}"] = _v(snap[k])
            m = f.get("entry/measurement")
            if m is not None:
                for k in m:
                    l = m.get(k, getlink=True)
                    if isinstance(l, h5py.ExternalLink):
                        r[f"link/{k}"] = os.path.basename(l.filename)
                        continue
                    d = m[k]
                    if isinstance(d, h5py.Dataset) and d.ndim == 1 and d.dtype.kind in "fiu" and len(d):
                        a = d[()]
                        r[f"m/{k}:n"], r[f"m/{k}:first"], r[f"m/{k}:last"] = len(a), a[0], a[-1]
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"[:200]
    return r


def tree(path):
    lines = []
    with h5py.File(path, "r") as f:
        def visit(g, pre):
            for k in g:
                l = g.get(k, getlink=True)
                if isinstance(l, h5py.ExternalLink):
                    lines.append(f"{pre}{k} -> {os.path.basename(l.filename)}:{l.path}")
                    continue
                o = g[k]
                if isinstance(o, h5py.Group):
                    lines.append(f"{pre}{k}/")
                    visit(o, pre + "  ")
                else:
                    lines.append(f"{pre}{k} {o.shape} {o.dtype}")
        visit(f, "")
    return "\n".join(lines)


def main(raw, out):
    global RAW, OUT
    RAW, OUT = raw, out
    names = sorted(n for n in os.listdir(RAW) if re.fullmatch(r"scan-\d+\.h5", n))
    with ProcessPoolExecutor(min(8, os.cpu_count() or 1), initializer=_init, initargs=(raw,)) as ex:
        rows = list(ex.map(one, names, chunksize=20))
    df = pd.DataFrame(rows).set_index("scan").sort_index()
    df.to_csv(os.path.join(OUT, "scan_table.csv"))
    motor = df.get("scan/motor1_name", pd.Series("?", index=df.index)).fillna("?")
    with open(os.path.join(OUT, "scan_trees.txt"), "w") as fh:
        for mo, g in df.groupby(motor):
            s = g.index[len(g) // 2]
            fh.write(f"===== motor1 = {mo}: {len(g)} Scans, Beispiel scan-{s}\n")
            try:
                fh.write(tree(os.path.join(RAW, f"scan-{s}.h5")) + "\n")
            except Exception as e:
                fh.write(f"Fehler {e}\n")
    extra = [p for p in ("which_sample.csv", "scan_daten.txt") if os.path.exists(os.path.join(RAW, "..", p))]
    with tarfile.open(os.path.join(OUT, "scan_table.tar.gz"), "w:gz") as t:
        for p in ("scan_table.csv", "scan_trees.txt"):
            t.add(os.path.join(OUT, p), arcname=p)
        for p in extra:
            t.add(os.path.join(RAW, "..", p), arcname=p)
    print(len(df), "Scans,", int(df.get("error", pd.Series(dtype=str)).notna().sum()), "Fehler;",
          "Motoren:", motor.value_counts().to_dict())
    print("fertig:", os.path.join(OUT, "scan_table.tar.gz"),
          round(os.path.getsize(os.path.join(OUT, "scan_table.tar.gz")) / 1e6, 2), "MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
