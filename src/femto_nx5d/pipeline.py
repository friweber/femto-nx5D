'''
FemtoMAX (MAX IV) -> nx5d.

Converts FemtoMAX Sardana scans (`scan-N.h5` + `scan-N_pilatus.h5` +
`scan-N_oscc_02_maui.h5`) into the canonical nx5d scan format (the same
layout the PXS import writes to `scans/rNNNN.h5`), seeds the spice of one
nx5d proposal per sample, and provides the recipes used by `.cooked`.

Layout produced (one proposal per sample, because the samples were measured
interleaved and each has its own geometry):

    <proposal_root>/<sample>/scans/<scan>.h5
    <proposal_root>/<sample>/spice/<uuid>

Usage from a notebook:

    from femto_nx5d import pipeline as fmx
    cfg = fmx.load_config("femtomax_config.json")
    cat = fmx.catalog(cfg)                         # reads master files only
    fmx.ingest_scan(123456, cfg)                   # -> <sample>/scans/123456.h5
    repo = fmx.repository(cfg)                     # nx5d.Repository + spice
    run = repo.proposal(key="SampleA").scan(key="123456")
    run.raw; run.cooked

Usage from a terminal (batch, resumable; config from ./femtomax_config.json or --config):

    femto-nx5d catalog
    femto-nx5d ingest --sample SampleA --type tRSS
    femto-nx5d ingest 123456 123457
'''

import json
import logging
import os
import re
import time
import warnings
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger("femtomax_nx5d")
# masked detector gaps give all-NaN rows/columns -> harmless numpy warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

INGEST_VERSION = "fmx-1"

# Sardana snapshot values copied per frame (name in /meta -> name in output)
SNAPSHOT_FIELDS = {
    "e_gon01_pit": None,     # handled separately (-> theta/ttheta)
    "e_gon01_x": "measurement/e_gon01_x",
    "e_gon01_y": "measurement/e_gon01_y",
    "e_gon01_z": "measurement/e_gon01_z",
    "temperature_a": "measurement/temperature_a",
    "temperature_b": "measurement/temperature_b",
    "l_lwp01_energy": "measurement/pump",
    "electron_bunch_charge": "measurement/bunch_charge",
    "xraydelay": None,       # handled separately (-> delay_set)
}


# ---------------------------------------------------------------------------
# configuration

def load_config(path="femtomax_config.json"):
    path = Path(path)
    cfg = json.loads(path.read_text())
    cfg["_config_dir"] = str(path.resolve().parent)
    cfg["_config_path"] = str(path.resolve())
    cfg["raw_dirs"] = [os.path.expanduser(d) for d in cfg["raw_dirs"]]
    cfg["proposal_root"] = os.path.expanduser(cfg["proposal_root"])
    return cfg


def _cfg_path(cfg, key):
    p = Path(cfg[key])
    return p if p.is_absolute() else Path(cfg["_config_dir"]) / p


_SKIP_ROOTS = {"proc", "sys", "dev", "usr", "lib", "lib32", "lib64", "bin", "sbin",
               "etc", "opt", "var", "run", "boot", "tmp", "snap"}


def locate_raw_dirs(beamtime="2025 Oct", subdirs=("raw2", "raw"),
                    roots=("/",), max_depth=5, max_dirs=50000):
    '''
    Searches the file system (directories only, depth-limited) for
    `.../FemtoMAX/<beamtime>/<subdir>` and returns the existing paths.
    Use it when `raw_dirs` in the config does not exist on this machine.
    '''
    found, visited = [], 0
    stack = [(Path(r), 0) for r in roots]
    while stack and visited < max_dirs:
        d, depth = stack.pop()
        visited += 1
        try:
            entries = [e for e in os.scandir(d) if e.is_dir(follow_symlinks=True)]
        except (PermissionError, FileNotFoundError, NotADirectoryError, OSError):
            continue
        for e in entries:
            if e.name.startswith(".") or (depth == 0 and e.name in _SKIP_ROOTS):
                continue
            p = Path(e.path)
            if e.name.lower() == "femtomax":
                for s in subdirs:
                    if (p / beamtime / s).is_dir():
                        found.append(str(p / beamtime / s))
                continue
            if depth + 1 < max_depth:
                stack.append((p, depth + 1))
    return found


def check_raw_dirs(cfg):
    '''
    Prints which `raw_dirs` exist. If none does, searches for the FemtoMAX
    folder and uses what it finds (for this session; put it into the config
    to make it permanent).
    '''
    ok = [d for d in cfg["raw_dirs"] if os.path.isdir(d)]
    for d in cfg["raw_dirs"]:
        print(f"{'OK   ' if os.path.isdir(d) else 'FEHLT'} {d}")
    if ok:
        return ok
    print("Keiner der raw_dirs existiert, suche FemtoMAX-Ordner ...")
    found = locate_raw_dirs()
    if not found:
        print("Nichts gefunden. Im Terminal pruefen, wo der Measurement-Share "
              "eingehaengt ist (z. B. `ls /`, `df -h`), und raw_dirs setzen.")
        return []
    print("Gefunden (fuer diese Sitzung verwendet; bitte in die Config eintragen):")
    for d in found:
        print("   ", d)
    cfg["raw_dirs"] = found
    return found


def find_master(scan, cfg):
    for d in cfg["raw_dirs"]:
        p = Path(d) / f"scan-{int(scan)}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(f"scan-{scan}.h5 not found in {cfg['raw_dirs']}")


def detect_sample(e_gon01_x, cfg, scan=None):
    """
    Sample of a scan: by the scan ranges in the config ("scan_ranges": [[first, last], ...])
    if a sample has them, otherwise the nearest sample position `e_gon01_x`.
    """
    if scan is not None:
        for name, s in cfg["samples"].items():
            if any(a <= int(scan) <= b for a, b in s.get("scan_ranges", [])):
                return name
        if any("scan_ranges" in s for s in cfg["samples"].values()):
            return None              # ranges given: scans outside them belong to no sample
    if e_gon01_x is None or not np.isfinite(e_gon01_x):
        return None
    name, s = min(cfg["samples"].items(),
                  key=lambda kv: abs(kv[1]["e_gon01_x"] - e_gon01_x))
    if abs(s["e_gon01_x"] - e_gon01_x) > cfg.get("sample_tolerance_mm", 1.0):
        return None
    return name


def _bin_edges(spec):
    '''[[start, stop, step], ...] -> concatenated bin edges'''
    if spec is None:
        return None
    edges = [np.arange(a, b + s / 2, s) for a, b, s in spec]
    return np.unique(np.concatenate(edges))


# ---------------------------------------------------------------------------
# master file reading (cheap: no detector data)

def _str(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def read_master(master):
    '''Scan definition and snapshot of one Sardana master file.'''
    with h5py.File(master, "r") as f:
        info = {
            "scan": int(_str(f["entry/entry_identifier"][()])),
            "command": _str(f["scan/command_line"][()]),
            "start_time": _str(f["entry/start_time"][()]),
            "ntriggers": int(f["scan/ntriggers"][()]),
            "motors": [],
            "points": {},
            "snapshot": {},
        }
        for i in range(1, 10):
            k = f"scan/motor{i}_name"
            if k not in f:
                break
            m = _str(f[k][()])
            info["motors"].append(m)
            if f"entry/measurement/{m}" in f:
                info["points"][m] = np.atleast_1d(f[f"entry/measurement/{m}"][()])
        for k in f.get("meta", {}):
            try:
                info["snapshot"][k] = float(f[f"meta/{k}"][()])
            except (TypeError, ValueError):
                pass
        for k in ("pcap_xray", "pcap_laser"):
            if f"entry/measurement/{k}" in f:
                info[k] = f[f"entry/measurement/{k}"][()].ravel()
    info["n_points"] = max([len(v) for v in info["points"].values()] or [0])
    return info


def scan_type(info, cfg):
    for m in info["motors"]:
        if m in cfg["scan_types"]:
            return cfg["scan_types"][m]
    return "other"


def catalog(cfg, refresh=False):
    '''
    Table of all scans in `raw_dirs` (master files only, ~40 KB each).
    Cached as `catalog_file`; pass `refresh=True` to rebuild.
    '''
    import pandas as pd
    out = _cfg_path(cfg, "catalog_file")
    if out.exists() and not refresh:
        df = pd.read_csv(out, index_col="scan")
        if set(df["raw_dir"].unique()) <= set(cfg["raw_dirs"]):
            return df
        print(f"{out.name} was built from other raw_dirs -> rebuilding")

    rows = []
    for d in cfg["raw_dirs"]:
        d = Path(d)
        if not d.exists():
            logger.warning(f"raw dir not found: {d}")
            continue
        for master in sorted(d.glob("scan-[0-9]*.h5")):
            if not re.fullmatch(r"scan-\d+\.h5", master.name):
                continue
            try:
                info = read_master(master)
            except Exception as e:
                logger.warning(f"{master.name}: {e}")
                continue
            snap = info["snapshot"]
            pil = master.with_name(master.stem + "_pilatus.h5")
            rows.append({
                "scan": info["scan"],
                "type": scan_type(info, cfg),
                "sample": detect_sample(snap.get("e_gon01_x"), cfg, info["scan"]),
                "command": info["command"],
                "n_points": info["n_points"],
                "ntriggers": info["ntriggers"],
                "start_time": info["start_time"],
                "temperature_a": snap.get("temperature_a"),
                "e_gon01_x": snap.get("e_gon01_x"),
                "e_gon01_pit": snap.get("e_gon01_pit"),
                "pump": snap.get("l_lwp01_energy"),
                "xraydelay_ps": snap.get("xraydelay", np.nan) * 1e12,
                "pilatus_GB": pil.stat().st_size / 1e9 if pil.exists() else 0.0,
                "raw_dir": str(d),
            })
    if not rows:
        raise FileNotFoundError(
            f"no scan-N.h5 master files found in raw_dirs={cfg['raw_dirs']}; "
            f"check the paths (fmx.check_raw_dirs(cfg) searches for them)")
    df = pd.DataFrame(rows).set_index("scan")
    df = df[~df.index.duplicated()].sort_index()  # first raw_dir wins
    df.to_csv(out)
    return df


# ---------------------------------------------------------------------------
# ingest: FemtoMAX scan -> nx5d scan file

def scan_file(cfg, sample, scan):
    return Path(cfg["proposal_root"]) / sample / "scans" / f"{int(scan)}.h5"


def ingest_scan(scan, cfg, sample=None, overwrite=False, block=32):
    '''
    Converts one FemtoMAX scan into `<proposal_root>/<sample>/scans/<scan>.h5`.

    Detector frames are streamed block-wise (never the full stack in memory),
    cut to the ROI, masked (pixels < 0 and the dilated module gaps -> NaN;
    xrayutilities' gridder ignores NaN) and stored gzip-compressed.

    Per frame: delay (ping-based, ps), delay_set (Sardana setpoint, ps),
    theta/ttheta (raw `e_gon01_pit`; offsets live in the spice), pump,
    temperatures, ping values, pcap flags.

    Returns the output path. Skips existing files of the same ingest version
    unless `overwrite=True`.
    '''
    t_start = time.time()
    if not overwrite:
        # already converted (for any sample)? -> skip without touching raw data
        for done in Path(cfg["proposal_root"]).glob(f"*/scans/{int(scan)}.h5"):
            if sample not in (None, done.parent.parent.name):
                continue
            with h5py.File(done, "r") as f:
                if f.attrs.get("fmx-ingest-version") == INGEST_VERSION:
                    logger.info(f"{scan}: exists, skipped ({done})")
                    return done
    master = find_master(scan, cfg)
    info = read_master(master)
    snap = info["snapshot"]
    sample = sample or detect_sample(snap.get("e_gon01_x"), cfg, scan)
    if sample is None:
        raise RuntimeError(f"{scan}: cannot assign a sample "
                           f"(e_gon01_x={snap.get('e_gon01_x')}); pass sample=...")
    stype = scan_type(info, cfg)

    out = scan_file(cfg, sample, scan)
    if out.exists() and not overwrite:
        with h5py.File(out, "r") as f:
            if f.attrs.get("fmx-ingest-version") == INGEST_VERSION:
                logger.info(f"{scan}: exists, skipped ({out})")
                return out

    pil_path = master.with_name(f"scan-{int(scan)}_pilatus.h5")
    osc_path = master.with_name(f"scan-{int(scan)}_oscc_02_maui.h5")

    # --- per-frame scalars -------------------------------------------------
    ntrig = info["ntriggers"]
    n_expected = info["n_points"] * ntrig if info["n_points"] else ntrig

    with h5py.File(pil_path, "r") as fp:
        n_pil = fp["entry/instrument/pilatus/data"].shape[0]

    ping = {}
    if osc_path.exists():
        with h5py.File(osc_path, "r") as fo:
            for ch in ("Ping_Ch2_value", "Ping_Ch4_value"):
                if f"data/{ch}" in fo:
                    ping[ch] = fo[f"data/{ch}"][()] / 1e-12  # s -> ps

    n = min([n_pil, n_expected] + [len(v) for v in ping.values()])
    if n == 0:
        raise RuntimeError(f"{scan}: no complete frames")
    if n < n_expected:
        logger.warning(f"{scan}: {n} of {n_expected} frames (aborted scan?)")

    def per_frame(name, scale=1.0):
        # scanned motor: one value per point, repeated per trigger;
        # otherwise the pre-scan snapshot value
        if name in info["points"]:
            return np.repeat(info["points"][name] * scale, ntrig)[:n]
        return np.full(n, snap.get(name, np.nan) * scale)

    pit = per_frame("e_gon01_pit")
    delay_set = per_frame("xraydelay", 1e12)
    pch = cfg.get("ping_channel", "Ping_Ch2_value")
    if pch in ping:
        delay = ping[pch][:n] - cfg["ping_t0_ps"]
    else:
        logger.warning(f"{scan}: no {pch}, using setpoint as delay")
        delay = delay_set.copy()

    # --- write --------------------------------------------------------------
    # Pilatus rows lie in the scattering plane. Images are stored transposed,
    # (index, out-of-plane, in-plane), as in the PXS files, so that with
    # imageAxes ("y-", "z-") xraydc labels the specular direction "qz".
    (r0, r1), (c0, c1) = cfg["roi"]["rows"], cfg["roi"]["cols"]
    nr, nc = r1 - r0, c1 - c0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.part")

    header = {
        "sample": sample,
        "command": info["command"],
        "start_time": info["start_time"],
        "temperature": round(snap.get("temperature_a", np.nan), 1),
        "pump": round(snap.get("l_lwp01_energy", np.nan), 2),
        "frames": n,
    }
    primary = {
        "tRSS": "measurement/delay",
        "sRSM": "instrument/motors/theta",
        "fRSS": "measurement/pump",
    }.get(stype)

    read_roi = _roi_reader(pil_path, (r0, r1, c0, c1), cfg.get("read_mode", "roi"))
    with h5py.File(tmp, "w") as fo, read_roi:
        fo.attrs["k-scan-type"] = stype
        fo.attrs["k-frame-guard"] = "instrument/pilatus/images"
        if primary:
            fo.attrs["k-primary:0"] = primary
        for k, v in header.items():
            fo.attrs[f"k-header:{k}"] = v
        fo.attrs["fmx-ingest-version"] = INGEST_VERSION
        fo.attrs["source"] = str(master)
        fo.attrs["roi"] = json.dumps(cfg["roi"])
        fo.attrs["ping_t0_ps"] = cfg["ping_t0_ps"]
        fo.attrs["ping_channel"] = pch

        img = fo.create_dataset(
            "instrument/pilatus/images", shape=(n, nc, nr), dtype="float32",
            chunks=(min(16, n), nc, nr), compression="gzip",
            compression_opts=4, shuffle=True)

        gap_mask = None
        max_count = cfg.get("mask", {}).get("max_count")
        for i in range(0, n, block):
            j = min(i + block, n)
            a = read_roi(i, j).astype("float32").transpose(0, 2, 1)
            if gap_mask is None:
                gap_mask = a[0] == -1
                if cfg.get("mask", {}).get("dilate_gaps", True):
                    from scipy.ndimage import binary_dilation
                    gap_mask = binary_dilation(gap_mask, structure=np.ones((3, 3)))
            a[:, gap_mask] = np.nan
            a[a < 0] = np.nan
            if max_count is not None:
                a[a > max_count] = np.nan
            img[i:j] = a
        fo.attrs["masked_pixels"] = int(gap_mask.sum())

        fo["instrument/motors/theta"] = pit
        # detector angle: follows pit (convention of the earlier FemtoMAX
        # notebooks) or stays at the pre-scan pit (detector fixed)
        if cfg.get("ttheta_follows_pit", True):
            fo["instrument/motors/ttheta"] = pit
        else:
            fo["instrument/motors/ttheta"] = np.full(n, snap["e_gon01_pit"])
        fo["measurement/delay"] = delay
        fo["measurement/delay"].attrs["units"] = "ps"
        fo["measurement/delay_set"] = delay_set
        fo["measurement/delay_set"].attrs["units"] = "ps"
        for ch, v in ping.items():
            fo[f"measurement/{ch.lower().replace('_value', '')}"] = v[:n]
        for name, dst in SNAPSHOT_FIELDS.items():
            if dst is not None:
                fo[dst] = per_frame(name)
        for k in ("pcap_xray", "pcap_laser"):
            if k in info and len(info[k]) >= n:
                fo[f"measurement/{k}"] = info[k][:n]
        fo["measurement/frame"] = np.arange(n)

    os.replace(tmp, out)
    dt = time.time() - t_start
    gb = pil_path.stat().st_size / 1e9
    logger.info(f"{scan}: {sample} {stype} {n} frames -> {out} "
                f"({dt:.0f} s, {gb:.1f} GB raw file, {gb / max(dt, 1e-3) * 1e3:.0f} MB/s effective)")
    return out


PILATUS_PATH = "entry/instrument/pilatus/data"


class _roi_reader:
    '''
    Reads ROI blocks `read(i, j)` -> array (j-i, rows, cols) from the Pilatus
    file. The raw frames are stored uncompressed, one chunk per frame, so
    only the needed bytes have to be read:

      "roi"      h5py with the chunk cache disabled: HDF5 reads only the ROI
                 pixels (row segments), ~0.16 MB instead of 4.9 MB per frame.
      "rowblock" reads the contiguous block of the ROI rows per frame
                 (~1.2 MB/frame) with one pread, cuts the columns in numpy.
      "full"     plain h5py (reads whole frames, 4.9 MB/frame).

    Which is fastest depends on the file system; see `benchmark_read`.
    '''

    def __init__(self, path, roi, mode="roi"):
        self.path, self.roi, self.mode = str(path), roi, mode
        if mode not in ("roi", "rowblock", "full"):
            raise ValueError(f"read_mode {mode!r}: use roi, rowblock or full")

    def __enter__(self):
        r0, r1, c0, c1 = self.roi
        if self.mode == "rowblock":
            with h5py.File(self.path, "r") as f:
                d = f[PILATUS_PATH]
                if d.compression is not None or d.chunks is None or d.chunks[0] != 1:
                    self.mode = "roi"   # layout not as expected -> safe path
                else:
                    _, self.h, self.w = d.shape
                    self.dtype = d.dtype
                    self.offsets = [d.id.get_chunk_info_by_coord((k, 0, 0)).byte_offset
                                    for k in range(d.shape[0])]
            if self.mode == "rowblock":
                self.fh = open(self.path, "rb", buffering=0)
                return self
        cache = 0 if self.mode == "roi" else 1024 ** 2
        self.f = h5py.File(self.path, "r", rdcc_nbytes=cache)
        self.d = self.f[PILATUS_PATH]
        return self

    def __call__(self, i, j):
        r0, r1, c0, c1 = self.roi
        if self.mode == "rowblock":
            it = self.dtype.itemsize
            out = np.empty((j - i, r1 - r0, c1 - c0), self.dtype)
            for k in range(i, j):
                buf = os.pread(self.fh.fileno(), (r1 - r0) * self.w * it,
                               self.offsets[k] + r0 * self.w * it)
                out[k - i] = np.frombuffer(buf, self.dtype).reshape(r1 - r0, self.w)[:, c0:c1]
            return out
        return self.d[i:j, r0:r1, c0:c1]

    def __exit__(self, *exc):
        if self.mode == "rowblock":
            self.fh.close()
        else:
            self.f.close()


def benchmark_read(cfg, scan, n_frames=60, parallel=(2, 4, 8)):
    '''
    Measures the read speed of the three read modes on this machine, each on
    a different block of frames of `scan` (so the OS cache does not help),
    then the fastest mode with several parallel processes (`parallel`).
    Use a long scan that has not been ingested yet, e.g. a fluence burst
    scan (fRSS, ~4600 frames), which the series ingest does not touch.
    '''
    master = find_master(scan, cfg)
    pil = master.with_name(f"scan-{int(scan)}_pilatus.h5")
    with h5py.File(pil, "r") as f:
        nmax = f[PILATUS_PATH].shape[0]
    roi = (*cfg["roi"]["rows"], *cfg["roi"]["cols"])
    if nmax < 3 * n_frames:
        n_frames = nmax // 3
    frame_mb = 831 * 1475 * 4 / 1e6
    res = {}
    for k, mode in enumerate(("full", "rowblock", "roi")):
        i = k * n_frames
        t = time.time()
        with _roi_reader(pil, roi, mode) as rd:
            rd(i, i + n_frames)
        dt = (time.time() - t) / n_frames
        res[mode] = dt
        print(f"{mode:9s}: {dt * 1e3:6.1f} ms/frame -> {frame_mb / dt:6.0f} MB/s effective, "
              f"1 TB raw in {1e6 / frame_mb * dt / 3600:5.1f} h")
    best = min(res, key=res.get)
    print(f"schnellster Modus: {best}  (in der Config: \"read_mode\": \"{best}\")")

    # parallel readers (like ingest --jobs N), each on its own block of frames
    print(f"CPU-Kerne: {_available_cpus()}")
    start = 3 * n_frames
    for jobs in parallel:
        if start + jobs * n_frames > nmax:
            print(f"jobs={jobs}: Scan zu kurz für den Test")
            continue
        from concurrent.futures import ProcessPoolExecutor
        args = [(str(pil), roi, best, start + q * n_frames, n_frames) for q in range(jobs)]
        t = time.time()
        with ProcessPoolExecutor(jobs) as ex:
            list(ex.map(_bench_block, args))
        dt = time.time() - t
        start += jobs * n_frames
        rate = jobs * n_frames * frame_mb / dt
        res[f"{best}_jobs{jobs}"] = dt / (jobs * n_frames)
        print(f"{best}, jobs={jobs}: {rate:6.0f} MB/s effective gesamt, "
              f"1 TB raw in {1e6 / rate / 3600:5.1f} h")
    return res


def _bench_block(args):
    path, roi, mode, i, n = args
    with _roi_reader(path, roi, mode) as rd:
        rd(i, i + n)


def _available_cpus():
    '''CPUs usable by this process (respects container limits).'''
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def _ingest_one(args):
    s, cfg, overwrite = args
    try:
        return s, ingest_scan(s, cfg, overwrite=overwrite)
    except Exception as e:
        return s, e


def ingest_many(scans, cfg, overwrite=False, jobs=1):
    '''
    Ingests a list of scans (existing ones are skipped); returns
    {scan: path or exception}. `jobs` > 1 runs several scans in parallel
    processes, which helps when reading over the network.
    '''
    scans = [int(s) for s in scans]
    todo = [(s, cfg, overwrite) for s in scans]
    res = {}
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            it = ex.map(_ingest_one, todo)
            for k, (s, r) in enumerate(it):
                res[s] = r
                print(f"[{k + 1}/{len(scans)}] {s}: "
                      f"{'FAILED ' + str(r) if isinstance(r, Exception) else 'ok'}", flush=True)
    else:
        for k, t in enumerate(todo):
            s, r = _ingest_one(t)
            res[s] = r
            print(f"[{k + 1}/{len(scans)}] {s}: "
                  f"{'FAILED ' + str(r) if isinstance(r, Exception) else 'ok'}", flush=True)
    return res


# ---------------------------------------------------------------------------
# nx5d repository + spice

def exp_info(sample_cfg, cfg):
    # stored images are (out-of-plane = ROI cols, in-plane = ROI rows)
    (r0, r1), (c0, c1) = cfg["roi"]["rows"], cfg["roi"]["cols"]
    ctr = sample_cfg["center"]
    return {
        "goniometerAxes": {"theta": "y-"},
        "detectorAxes": {"ttheta": "y-"},
        "detectorTARAlign": [0.0, 0.0, 0.0],
        "imageAxes": ["y-", "z-"],
        "imageSize": [c1 - c0, r1 - r0],
        "imageCenter": [ctr["col"], ctr["row"]],
        "imageDistance": sample_cfg["imageDistance"],
        "imageChannelSize": [0.172, 0.172],
        "sampleFaceUp": "z+",
        "beamDirection": [1, 0, 0],
        "sampleNormal": [0, 0, 1],
        "beamEnergy": sample_cfg["beamEnergy"],
    }


def seed_spice(proposal, cfg):
    '''
    Seeds the spice of one sample proposal (idempotent: existing types are
    left untouched, so later `spice.update(...)` revisions are kept).

    Safe against several notebooks/processes seeding at the same time: each
    seed gets a fixed UUID derived from (sample, type) and is created
    exclusively, so a second writer fails instead of writing a duplicate
    (which nx5d would report as SpiceRevisionConflict).
    '''
    import uuid
    s = cfg["samples"][proposal.key]
    proposal.wait_for_scans()  # spice is attached by the scan-loading thread
    sp = proposal.spice
    seeds = {
        "exp_info": exp_info(s, cfg),
        "offsets": {"theta": s["theta_offset"], "ttheta": s["ttheta_offset"], "delay": 0.0},
        "recipes": {"tRSS": "pymod:///femtomax_nx5d#tRSS",
                    "fRSS": "pymod:///femtomax_nx5d#fRSS",
                    "sRSM": "pymod:///femtomax_nx5d#sRSM"},
        "qargs": {"groupby": None, "dims": None, "qsize": None, "gridargs": None},
        "binning": {"delay": cfg.get("delay_bins_ps"), "pump_round": 1},
    }
    written = False
    for handle, payload in seeds.items():
        if handle in sp.collection.handles():
            continue
        obj = {"version": "kmc3/1a", "type": handle, "handle": handle,
               "uuid": uuid.uuid5(uuid.NAMESPACE_URL, f"femtomax-seed/{proposal.key}/{handle}"),
               "revision": 1, "anchor": "", "data": payload}
        try:
            sp.save(obj)
        except FileExistsError:
            pass                 # seeded concurrently by another process
        written = True
    if written:                  # pick up seeds written by other processes
        sp._collection = sp.init_collection()
    return sp


def repair_spice(cfg, sample, dry_run=True):
    '''
    Finds duplicated spice objects (same type, anchor, revision and identical
    data, e.g. from two notebooks seeding at once) and moves all but the
    oldest to `<sample>/spice_duplicates/`. Objects that conflict with
    *different* data are only reported. Returns the list of moved files.
    '''
    import collections
    import shutil
    d = Path(cfg["proposal_root"]) / sample / "spice"
    groups = collections.defaultdict(list)
    for f in d.iterdir():
        if f.is_file():
            o = json.loads(f.read_text())
            groups[(o["handle"], o["anchor"], o["revision"])].append(
                (f.stat().st_mtime, f, json.dumps(o["data"], sort_keys=True)))
    moved = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x[0])
        if len({x[2] for x in items}) > 1:
            print(f"{sample} {key}: {len(items)} revisions with DIFFERENT data -> "
                  f"by hand: {[x[1].name for x in items]}")
            continue
        for _, f, _ in items[1:]:
            print(f"{sample} {key}: duplicate {f.name}" + (" (dry run)" if dry_run else " -> spice_duplicates/"))
            if not dry_run:
                (d.parent / "spice_duplicates").mkdir(exist_ok=True)
                shutil.move(str(f), str(d.parent / "spice_duplicates" / f.name))
            moved.append(f)
    return moved


def repository(cfg, seed=True):
    '''nx5d Repository over `<proposal_root>/<sample>/scans/<scan>.h5`.'''
    from nx5d import Repository
    root = Path(cfg["proposal_root"]).resolve()
    repo = Repository(
        url=f"file://{root}/{{proposal}}/scans/{{scan}}.h5",
        spice_url=f"file://{root}/{{proposal}}/spice",
    )
    if seed:
        for key in repo.all():
            if key in cfg["samples"]:
                seed_spice(repo.proposal(key=key), cfg)
    return repo


# ---------------------------------------------------------------------------
# recipes (referenced from the "recipes" spice as pymod:///femtomax_nx5d#...)

def _apply_offsets(data, spice):
    data = data.copy()
    for k, v in (spice.get("offsets") or {}).items():
        if k in data and v:
            data[k] = data[k] + v
    spice = dict(spice)
    spice["offsets"] = {}
    return data, spice


def _grouped_qmap(stype, data, groupby, default_dims, **spice):
    from xraydc.cooking import _groupby_rsm
    qargs = dict(spice.get("qargs") or {})
    qargs["groupby"] = groupby
    if qargs.get("dims") in (None, (), []):
        qargs["dims"] = default_dims
    spice["qargs"] = qargs
    return _groupby_rsm(stype, data, **spice)


def tRSS(stype, data, **spice):
    '''
    Time-resolved reciprocal space slice: frames are binned on the measured
    (ping) delay with the edges from spice "binning.delay", then each bin is
    mapped to (qy, qz) (override with qargs.dims). Adds `frames` per bin.
    '''
    data, spice = _apply_offsets(data.compute(), spice)
    edges = _bin_edges((spice.get("binning") or {}).get("delay"))
    if edges is not None:
        idx = np.digitize(data["delay"].values, edges) - 1
        ok = (idx >= 0) & (idx < len(edges) - 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        data = data.isel(index=np.where(ok)[0])
        data["delay"] = ("index", centers[idx[ok]])
    counts = data["delay"].groupby(data["delay"]).count()
    res = _grouped_qmap(stype, data, "delay", ("qy", "qz"), **spice)
    res["frames"] = ("delay", counts.values)
    return res


def fRSS(stype, data, **spice):
    '''Fluence series: grouped by rounded pump energy, mapped to (qy, qz).'''
    data, spice = _apply_offsets(data.compute(), spice)
    nd = (spice.get("binning") or {}).get("pump_round", 1)
    data["pump"] = data["pump"].round(nd)
    return _grouped_qmap(stype, data, "pump", ("qy", "qz"), **spice)


def sRSM(stype, data, **spice):
    '''
    Static RSM (pit scan): frames at the same angle are averaged first, then
    the stack is mapped to 3D (override with qargs.dims / qargs.qsize).
    '''
    from xraydc.cooking import sRSM as _sRSM
    data, spice = _apply_offsets(data.compute(), spice)
    data = data.set_coords("theta").groupby("theta").mean().rename({"theta": "index"})
    data["theta"] = ("index", data["index"].values)
    data = data.drop_vars("index")
    return _sRSM(stype, data, **spice)


# ---------------------------------------------------------------------------
# analysis helpers (operate on .cooked of tRSS / fRSS scans)

def qz_profile(cooked, qy=None, qz=None):
    '''Line profile along qz (sum over qy within `qy=(min, max)`).'''
    im = cooked["images"]
    if qy is not None:
        im = im.sel(qy=slice(*qy))
    if qz is not None:
        im = im.sel(qz=slice(*qz))
    return im.sum("qy")


def _gauss(x, a, x0, s, c):
    return a * np.exp(-0.5 * ((x - x0) / s) ** 2) + c


def fit_peak(profile, dim="qz"):
    '''
    Gaussian + constant fit of every 1D profile in `profile` (e.g. one per
    delay). Returns an xarray.Dataset with center, sigma, amplitude, the
    centre of mass `com` and fit errors.
    '''
    import xarray as xr
    from scipy.optimize import curve_fit
    x = profile[dim].values
    other = [d for d in profile.dims if d != dim]
    stacked = profile.transpose(*other, dim).values.reshape(-1, len(x))
    res = {k: [] for k in ("center", "center_err", "sigma", "amplitude", "com")}
    for y in stacked:
        y = np.nan_to_num(y)
        com = (x * y).sum() / y.sum() if y.sum() > 0 else np.nan
        p0 = (y.max() - y.min(), x[y.argmax()], (x[-1] - x[0]) / 20, y.min())
        try:
            p, cov = curve_fit(_gauss, x, y, p0=p0, maxfev=5000)
            err = np.sqrt(np.diag(cov))
        except Exception:
            p, err = [np.nan] * 4, [np.nan] * 4
        res["center"].append(p[1]); res["center_err"].append(err[1])
        res["sigma"].append(abs(p[2])); res["amplitude"].append(p[0])
        res["com"].append(com)
    shape = [profile.sizes[d] for d in other]
    coords = {d: profile[d] for d in other}
    return xr.Dataset({k: (other, np.array(v).reshape(shape)) for k, v in res.items()},
                      coords=coords)


def strain(peaks, ref=None, key="center", dim=None):
    '''
    Strain (in permille) from peak positions: -(q - q0)/q0 * 1e3.
    `ref`: q0 as number, a window (min, max) along `dim` to average over,
    or None for the first point. `dim` defaults to the first dimension
    (`delay` for tRSS, `pump` for fRSS).
    '''
    q = peaks[key]
    dim = dim or q.dims[0]
    first = q.dropna(dim).isel({dim: 0})
    if ref is None:
        q0 = first
    elif np.isscalar(ref):
        q0 = ref
    else:
        sel = q.sel({dim: slice(*ref)}).dropna(dim)
        # empty reference window (e.g. no negative delays) -> first valid point
        q0 = sel.mean(dim) if sel.sizes[dim] else first
    return -(q - q0) / q0 * 1e3


# ---------------------------------------------------------------------------
# measurement series

def series_table(cat, sample, stype="tRSS", t_step=10, x_step=0.1, pump_round=1):
    '''
    Scans of one sample and type with the series keys: `T` (temperature_a
    rounded to `t_step` K), `x` (sample position rounded to `x_step` mm,
    separates measurement spots) and `pump_r` (rounded pump energy).
    '''
    t = cat[(cat["sample"] == sample) & (cat["type"] == stype)].copy()
    t["T"] = ((t["temperature_a"] / t_step).round() * t_step).fillna(-1)   # -1 = no temperature recorded
    t["x"] = (t["e_gon01_x"] / x_step).round() * x_step
    t["pump_r"] = t["pump"].round(pump_round)
    return t


def fluence_series(cat, sample, min_fluences=4, **kw):
    '''
    {(T, x): {pump: [scans]}} for every temperature (and sample spot) with at
    least `min_fluences` different pump energies. Scans with the same pump
    (different delay ranges, repetitions) are later merged into one curve.
    '''
    t = series_table(cat, sample, **kw)
    out = {}
    for (T, x), g in t.groupby(["T", "x"]):
        if g["pump_r"].nunique() >= min_fluences:
            out[(float(T), round(float(x), 2))] = {
                float(p): [int(s) for s in gg.index] for p, gg in g.groupby("pump_r")}
    return out


def temperature_series(cat, sample, min_temperatures=5, **kw):
    '''{(pump, x): {T: [scans]}} for pump energies measured at many temperatures.'''
    t = series_table(cat, sample, **kw)
    out = {}
    for (p, x), g in t.groupby(["pump_r", "x"]):
        if g["T"].nunique() >= min_temperatures:
            out[(float(p), round(float(x), 2))] = {
                float(T): [int(s) for s in gg.index] for T, gg in g.groupby("T")}
    return out


def series_overview(cat, sample, **kw):
    '''Number of tRSS scans per (T, x) and pump energy -- to choose series.'''
    t = series_table(cat, sample, **kw)
    return t.groupby(["T", "x", "pump_r"]).size().unstack(fill_value=0)


def series_scans(series):
    '''All scan numbers of a series dict (for ingesting).'''
    return sorted({s for by in series.values() for scans in by.values() for s in scans})


def rsm_scans(cat, sample):
    """Angle scans of a sample, static ones (x-ray delay <= 0) first."""
    r = cat[(cat["sample"] == sample) & (cat["type"] == "sRSM")]
    key = r["xraydelay_ps"].fillna(0) if "xraydelay_ps" in r else 0
    return [int(s) for s in r.assign(_k=(key > 0)).sort_values(["_k"]).index]


def ingest_order(cat, sample, series="fluence"):
    '''
    Scans to ingest for one sample: RSM first, then the series from the
    smallest to the largest (by Pilatus GB), so that first complete series
    are available early.
    '''
    groups = []
    if series in ("fluence", "all"):
        groups += list(fluence_series(cat, sample).values())
    if series in ("temperature", "all"):
        groups += list(temperature_series(cat, sample).values())
    gb = cat["pilatus_GB"]
    groups = [sorted({s for v in by.values() for s in v}) for by in groups]
    groups.sort(key=lambda g: gb.reindex(g).sum())
    order = rsm_scans(cat, sample)[:1] + [s for g in groups for s in g]   # one static RSM
    return list(dict.fromkeys(order))


# ---------------------------------------------------------------------------
# background ingest (survives closing the notebook)

def _ingest_files(cfg, sample):
    log = Path(cfg["_config_dir"]) / f"ingest_{sample}.log"
    return log, log.with_suffix(".pid")


def _running_pid(cfg, sample):
    _, pidfile = _ingest_files(cfg, sample)
    if not pidfile.exists():
        return None
    pid = int(pidfile.read_text().strip() or 0)
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        state = [l for l in Path(f"/proc/{pid}/status").read_text().splitlines()
                 if l.startswith("State:")][0]
    except (OSError, IndexError):
        return None
    if b"femtomax_nx5d" not in cmd or "zombie" in state:
        return None
    return pid


def start_background_ingest(cfg, sample, series="fluence", jobs=3):
    '''
    Starts `femtomax_nx5d.py ingest --sample <sample> --series <series>` as a
    detached process (keeps running when the notebook is closed; stops only
    if the JupyterHub server itself is shut down). Does nothing if an ingest
    for this sample is already running. Log: ingest_<sample>.log next to
    the config.
    '''
    import subprocess
    import sys
    pid = _running_pid(cfg, sample)
    log, pidfile = _ingest_files(cfg, sample)
    if pid:
        print(f"Ingest {sample} läuft bereits (pid {pid}), Log: {log}")
        return pid
    cmd = [sys.executable, str(Path(__file__).resolve()), "--config", cfg["_config_path"],
           "ingest", "--sample", sample, "--series", series, "--jobs", str(jobs)]
    with open(log, "a") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True, cwd=cfg["_config_dir"])
    pidfile.write_text(str(p.pid))
    print(f"Ingest {sample} gestartet (pid {p.pid}), Log: {log}")
    return p.pid


def stop_background_ingest(cfg, sample):
    import signal
    pid = _running_pid(cfg, sample)
    if pid:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"Ingest {sample} (pid {pid}) gestoppt")


def ingest_status(cfg, cat, sample, scans):
    '''Prints how many of `scans` are converted, and the last log line.'''
    done = [s for s in scans if scan_file(cfg, sample, s).exists()]
    gb = cat["pilatus_GB"]
    pid = _running_pid(cfg, sample)
    print(f"{sample}: {len(done)}/{len(scans)} Scans umgewandelt "
          f"({gb.reindex(done).sum():.0f}/{gb.reindex(scans).sum():.0f} GB), "
          f"Ingest {'läuft (pid %d)' % pid if pid else 'läuft nicht'}")
    log, _ = _ingest_files(cfg, sample)
    if log.exists():
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        if lines:
            print("   letzte Logzeile:", lines[-1][:200])
    return done


def open_proposal(repo, sample):
    '''The nx5d proposal of `sample`, or None (with a hint) if nothing is converted yet.'''
    if sample not in repo.all():
        print(f"Für {sample} ist noch kein Scan umgewandelt -> Abschnitt 2 (Ingest) "
              f"starten und diese Zelle später erneut ausführen.")
        return None
    return repo.proposal(key=sample)


def cook_merged(proposal, scans, warn=True):
    '''
    Merges the raw frames of several scans (same sample, same scan type, e.g.
    different delay ranges at one fluence) and cooks them in one go, with the
    spice valid for the first scan. Scans not (yet) ingested are skipped.
    Returns None if none is available.
    '''
    import xarray as xr
    runs, missing = [], []
    for s in scans:
        if str(int(s)) in proposal.all():
            runs.append(proposal.scan(key=str(int(s))))
        else:
            missing.append(int(s))
    if missing and warn:
        logger.warning(f"not ingested (skipped): {missing}")
    if not runs:
        return None
    raws = [r.raw.compute() for r in runs]
    common = sorted(set.intersection(*[set(r.data_vars) for r in raws]))
    data = xr.concat([r[common] for r in raws], dim="index",
                     combine_attrs="drop_conflicts")
    spice = {k: v.data() for k, v in runs[0].spice.data.items()}
    recipe = runs[0]._load_recipe(spice["recipes"][runs[0].type])
    res = recipe(runs[0].type, data, **spice)
    res.attrs["scans"] = [int(r.key) for r in runs]
    for r in runs:
        _release(r)
    return res


def split_loops(scans):
    """Consecutive scan numbers form one measurement loop (repetition of the delay ranges)."""
    scans = sorted(int(s) for s in scans)
    loops = [[scans[0]]] if scans else []
    for s in scans[1:]:
        if s == loops[-1][-1] + 1:
            loops[-1].append(s)
        else:
            loops.append([s])
    return loops


def load_frames(proposal, scans, delay_tol_ps=3.0):
    """
    Raw frames of several scans as one dataset (variable 'scan' per frame).
    Frames whose ping delay deviates more than `delay_tol_ps` from the median
    of their set point (same scan, same delay_set) are dropped (ping glitches).
    Returns (data, spice of the first scan, number of dropped frames) or (None, None, 0).
    """
    import xarray as xr
    runs = [proposal.scan(key=str(s)) for s in scans if str(int(s)) in proposal.all()]
    if not runs:
        return None, None, 0
    parts, dropped = [], 0
    for r in runs:
        d = r.raw.compute()
        dl, ds_ = d["delay"].values, d["delay_set"].values
        ok = np.ones(dl.size, bool)
        for v in np.unique(ds_):
            sel = ds_ == v
            ok[sel] = np.abs(dl[sel] - np.nanmedian(dl[sel])) < delay_tol_ps
        dropped += int((~ok).sum())
        d = d.isel(index=np.where(ok)[0])
        d["scan"] = ("index", np.full(d.sizes["index"], int(r.key)))
        parts.append(d)
        _release(r)
    common = sorted(set.intersection(*[set(x.data_vars) for x in parts]))
    data = xr.concat([x[common] for x in parts], dim="index", combine_attrs="drop_conflicts")
    spice = {k: v.data() for k, v in runs[0].spice.data.items()}
    return data, spice, dropped


def _release(run):
    """nx5d keeps the fully loaded raw data (and cooked data) of every scan object
    in memory -> drop it after use, otherwise hundreds of scans fill the RAM."""
    run.__dict__.pop("_raw", None)
    run.__dict__.pop("_cooked", None)


def _mask_hot_pixels(d, factor=8.0):
    """Masks pixels whose time-averaged intensity is far above their 5x5
    neighbourhood (hot/defect pixels; they show up as static spikes in the profiles)."""
    from scipy.ndimage import median_filter
    img = d["images"].values
    with np.errstate(all="ignore"):
        mean = np.nanmean(img, axis=0)
    med = median_filter(np.nan_to_num(mean), size=5)
    noise = np.nanstd(mean) * 0.05 + 1e-6
    hot = np.isfinite(mean) & (mean > factor * med + 5 * noise) & (mean > np.nanpercentile(mean, 50))
    # keep the Bragg peak itself: a pixel is only hot if it is isolated (few hot neighbours)
    hot &= median_filter(hot.astype(np.uint8), size=3) == 0
    if hot.any():
        img = img.copy()
        img[:, hot] = np.nan
        d = d.assign(images=(d["images"].dims, img))
    return d


def load_profiles(proposal, scans, delay_tol_ps=3.0):
    """
    Memory-friendly variant of load_frames for the 'q_x = q_y = 0' analysis:
    every scan is loaded, reduced to row/column profiles (frame x ~200 values)
    and released. Returns dict(delay, prof, colp, qz, qy, dropped, spice) or None.
    """
    runs = [proposal.scan(key=str(s)) for s in scans if str(int(s)) in proposal.all()]
    if not runs:
        return None
    spice = {k: v.data() for k, v in runs[0].spice.data.items()}
    delays, profs, cols, dropped, qz_ref, qy = [], [], [], 0, None, None
    for r in runs:
        d = r.raw.compute()
        dl, ds_ = d["delay"].values, d["delay_set"].values
        ok = np.ones(dl.size, bool)
        for v in np.unique(ds_):
            sel = ds_ == v
            ok[sel] = np.abs(dl[sel] - np.nanmedian(dl[sel])) < delay_tol_ps
        dropped += int((~ok).sum())
        d = d.isel(index=np.where(ok)[0])
        d = _mask_hot_pixels(d)
        prof, qz, colp, qy = frame_profiles(d, spice)
        if qz_ref is None:
            qz_ref = qz
        elif not np.allclose(qz, qz_ref):
            prof = np.array([np.interp(qz_ref, qz, p, left=np.nan, right=np.nan) for p in prof])
        delays.append(d["delay"].values)
        profs.append(prof.astype(np.float32))
        cols.append(colp.astype(np.float32))
        del d
        _release(r)
    return {"delay": np.concatenate(delays), "prof": np.concatenate(profs), "colp": np.concatenate(cols),
            "qz": qz_ref, "qy": qy, "dropped": dropped, "spice": spice}


def frame_profiles(data, spice):
    """
    'q_x = q_y = 0' profiles: every in-plane detector row gets
    q_z = 4 pi / lambda * sin(2theta / 2) from its scattering angle (the sample
    angle theta does not enter); intensities are averaged over the
    out-of-plane columns. Returns (profiles (frames, rows), qz (rows),
    column profiles (frames, cols), qy (cols)).
    """
    ei, off = spice["exp_info"], spice.get("offsets") or {}
    img = data["images"].values                             # (frames, out-of-plane, in-plane), float32
    lam = 12398.42 / ei["beamEnergy"]                        # Angstrom
    col_c, row_c = ei["imageCenter"]
    dist = ei["imageDistance"]
    px_col, px_row = ei["imageChannelSize"]
    tth = data["ttheta"].values + off.get("ttheta", 0.0)
    rows = np.arange(img.shape[2])
    d2 = np.degrees(np.arctan((rows - row_c) * px_row / dist))
    with np.errstate(all="ignore"):
        prof = np.nanmean(img, axis=1)                      # (frames, rows)
        colp = np.nanmean(img, axis=2)                      # (frames, cols)
    qz = 4 * np.pi / lam * np.sin(np.radians(tth[0] - d2) / 2)
    if np.ptp(tth) > 1e-4:                                   # different detector angles -> common grid
        for i in range(prof.shape[0]):
            q_i = 4 * np.pi / lam * np.sin(np.radians(tth[i] - d2) / 2)
            prof[i] = np.interp(qz, q_i[::-1], prof[i][::-1], left=np.nan, right=np.nan)
    qy = 2 * np.pi / lam * np.sin(np.arctan((np.arange(img.shape[1]) - col_c) * px_col / dist))
    order = np.argsort(qz)
    return prof[:, order], qz[order], colp, qy


def _bin(values, edges):
    idx = np.digitize(values, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    return idx, ok


def fit_profiles(qz, P, center, halfwidth):
    """
    Gaussian + constant fit of every row of P (delay, qz) inside
    center +- halfwidth, with bounds; fits that hit a bound or have an error
    larger than the window are set to NaN. Returns dict of arrays.
    """
    from scipy.optimize import curve_fit
    sel = np.abs(qz - center) <= halfwidth
    x = qz[sel]
    out = {k: np.full(P.shape[0], np.nan) for k in ("center", "center_err", "sigma", "amplitude", "com")}
    for i, y in enumerate(P[:, sel]):
        m = np.isfinite(y)
        if m.sum() < 6:
            continue
        xx, yy = x[m], y[m]
        base = np.nanmin(yy)
        w = np.clip(yy - base, 0, None)
        if w.sum() > 0:
            out["com"][i] = (xx * w).sum() / w.sum()
        p0 = (yy.max() - base, xx[np.argmax(yy)], halfwidth / 4, base)
        lo = (0, center - halfwidth, halfwidth / 40, -np.inf)
        hi = (np.inf, center + halfwidth, halfwidth, np.inf)
        try:
            p, cov = curve_fit(_gauss, xx, yy, p0=p0, bounds=(lo, hi), maxfev=4000)
            err = np.sqrt(np.diag(cov))
        except Exception:
            continue
        at_bound = (np.isclose(p[2], lo[2], rtol=1e-3) or np.isclose(p[2], hi[2], rtol=1e-3) or
                    np.isclose(p[1], lo[1], atol=1e-6) or np.isclose(p[1], hi[1], atol=1e-6))
        if at_bound or not np.isfinite(err[1]) or err[1] > halfwidth / 4:
            continue
        out["center"][i], out["center_err"][i] = p[1], err[1]
        out["sigma"][i], out["amplitude"][i] = abs(p[2]), p[0]
    return out


def _loop_curve(proposal, scans, cfg, qz_center, qx_qy_zero=True, min_frames=5, delay_tol_ps=3.0):
    """Binned profiles + fits of one loop. Returns dict or None."""
    if qx_qy_zero:
        L = load_profiles(proposal, scans, delay_tol_ps)
        if L is None:
            return None
        spice, dropped = L["spice"], L["dropped"]
        prof, qz, colp, qy, raw_delay = L["prof"], L["qz"], L["colp"], L["qy"], L["delay"]
    else:
        data, spice, dropped = load_frames(proposal, scans, delay_tol_ps)
        if data is None:
            return None
        raw_delay = data["delay"].values
    hw = cfg.get("qz_halfwidth", 0.04)
    ref = tuple(cfg.get("strain_ref_ps", (-20, -2)))
    edges = _bin_edges((spice.get("binning") or {}).get("delay") or cfg.get("delay_bins_ps"))
    delays = raw_delay + ((spice.get("offsets") or {}).get("delay") or 0.0)
    idx, ok = _bin(delays, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if qx_qy_zero:
        nb = len(edges) - 1
        counts = np.bincount(idx[ok], minlength=nb)
        keep = np.where(counts >= min_frames)[0]
        P = np.full((keep.size, qz.size), np.nan)
        C = np.full((keep.size, qy.size), np.nan)
        with np.errstate(all="ignore"):
            for j, b in enumerate(keep):
                fr = np.where(ok & (idx == b))[0]
                P[j] = np.nanmean(prof[fr], axis=0)
                C[j] = np.nanmean(colp[fr], axis=0)
        bd, counts = centers[keep], counts[keep]
    else:
        runs = [proposal.scan(key=str(s)) for s in scans if str(int(s)) in proposal.all()]
        recipe = runs[0]._load_recipe(spice["recipes"][runs[0].type])
        ck = recipe(runs[0].type, data, **spice)
        prof2 = qz_profile(ck)
        keep = ck["frames"].values >= min_frames
        P, qz = prof2.values[keep], prof2.qz.values
        bd, counts = ck.delay.values[keep], ck["frames"].values[keep]
        qy, C = ck.qy.values, ck["images"].sum("qz").values[keep]
    if bd.size == 0:
        return None
    # peak window from the mean profile (near qz_center if given)
    mean = np.nanmean(P, axis=0)
    search = np.abs(qz - qz_center) <= 5 * hw if qz_center is not None else np.isfinite(mean)
    q0 = qz[search][np.nanargmax(mean[search])]
    fit = fit_profiles(qz, P, q0, hw)
    # peak position perpendicular (snail trace): column COM around the peak
    base = np.nanmin(C, axis=1, keepdims=True)
    w = np.clip(C - base, 0, None)
    wsum = np.nansum(w, axis=1)
    qy_com = np.nansum(w * qy, axis=1) / np.where(wsum > 0, wsum, np.nan)
    # profiles on a grid relative to the window centre (for the waterfall plot)
    rel = np.linspace(-2 * hw, 2 * hw, 81)
    Pw = np.array([np.interp(q0 + rel, qz, p, left=np.nan, right=np.nan) for p in P])
    refsel = (bd >= ref[0]) & (bd <= ref[1]) & np.isfinite(fit["center"])
    return {"delay": bd, "frames": counts, "qy_com": qy_com, "profile": Pw, "q_rel": rel, "q_window": q0,
            "q0": float(np.nanmean(fit["center"][refsel])) if refsel.any() else np.nan,
            "dropped": dropped, "scans": list(scans), **fit}


def analyze_curve(proposal, scans, cfg, qz_center=None, qx_qy_zero=True, drift_correction=True,
                  min_frames=5, delay_tol_ps=3.0):
    """
    One curve (e.g. one fluence) from several scans: the scans are split into
    loops (repetitions), every loop is binned and fitted on its own, shifted
    to the common pre-t0 peak position (drift correction) and the loops are
    averaged (weights = frames). Returns an xarray.Dataset over delay.
    """
    import xarray as xr
    loops = [l for l in (_loop_curve(proposal, lp, cfg, qz_center, qx_qy_zero, min_frames, delay_tol_ps)
                         for lp in split_loops(scans)) if l is not None]
    if not loops:
        return None
    q0s = np.array([l["q0"] for l in loops])
    q_ref = np.nanmean(q0s) if np.isfinite(q0s).any() else np.nan
    allb = np.unique(np.concatenate([l["delay"] for l in loops]))
    acc = {k: np.zeros(allb.size) for k in ("w", "c", "s", "a", "m", "qy", "inv", "f")}
    prof = np.zeros((allb.size, loops[0]["profile"].shape[1]))

    def onto(x, y, valid):
        """Interpolates a loop curve onto the common delay grid inside the loop's
        measured range (no extrapolation). The grid only has bins where some loop
        has data, so real gaps (no data at all) stay gaps."""
        xv, yv = x[valid], y[valid]
        out = np.full(allb.size, np.nan)
        if xv.size < 2:
            return out
        inside = (allb >= xv[0]) & (allb <= xv[-1])
        out[inside] = np.interp(allb[inside], xv, yv)
        return out

    # Each loop is interpolated onto all bins before averaging: with slightly
    # different ping delays, loops would otherwise fill alternate bins (zigzag).
    for l in loops:
        shift = (q_ref - l["q0"]) if (drift_correction and np.isfinite(l["q0"])) else 0.0
        good = np.isfinite(l["center"])
        c = onto(l["delay"], l["center"] + shift, good)
        ok = np.isfinite(c)
        # constant weight per loop (its mean frames per bin): bin-to-bin varying
        # weights would let the average alternate between loops (zigzag)
        w = np.where(ok, float(np.mean(l["frames"][good])) if good.any() else 0.0, 0.0)
        acc["w"] += w
        acc["c"] += np.where(ok, c * w, 0)
        acc["s"] += np.nan_to_num(onto(l["delay"], l["sigma"], good) * w)
        acc["a"] += np.nan_to_num(onto(l["delay"], l["amplitude"], good) * w)
        acc["m"] += np.nan_to_num(onto(l["delay"], l["com"] + shift, good & np.isfinite(l["com"])) * w)
        acc["qy"] += np.nan_to_num(onto(l["delay"], l["qy_com"], good & np.isfinite(l["qy_com"])) * w)
        acc["inv"] += np.nan_to_num(onto(l["delay"], 1 / l["center_err"] ** 2, good))
        acc["f"] += w
        for k in range(prof.shape[1]):
            colk = l["profile"][:, k]
            prof[:, k] += np.nan_to_num(onto(l["delay"], colk, good & np.isfinite(colk)) * w)
    with np.errstate(all="ignore"):
        W = np.where(acc["w"] > 0, acc["w"], np.nan)
        ds = xr.Dataset(
            {"center": ("delay", acc["c"] / W), "center_err": ("delay", 1 / np.sqrt(acc["inv"])),
             "sigma": ("delay", acc["s"] / W), "amplitude": ("delay", acc["a"] / W),
             "com": ("delay", acc["m"] / W), "qy_com": ("delay", acc["qy"] / W),
             "frames": ("delay", acc["f"]),
             "profile": (("delay", "q_rel"), prof / np.where(acc["f"] > 0, acc["f"], np.nan)[:, None])},
            coords={"delay": allb, "q_rel": loops[0]["q_rel"]})
    ds["center_err"] = ds["center_err"].where(np.isfinite(ds["center_err"]))
    # every loop on its own (strain relative to its own pre-t0 position) for diagnostics
    sl = []
    for l in loops:
        good = np.isfinite(l["center"])
        q0 = l["q0"] if np.isfinite(l["q0"]) else q_ref
        sl.append(onto(l["delay"], -(l["center"] - q0) / q0 * 1e3, good))
    ds["strain_loop"] = (("loop", "delay"), np.array(sl))
    ds = ds.assign_coords(loop=np.arange(len(loops)))
    l0 = loops[0]
    shift0 = (q_ref - l0["q0"]) if (drift_correction and np.isfinite(l0["q0"])) else 0.0
    ds["q_window"] = l0["q_window"] + shift0     # absolute q of q_rel = 0 (drift corrected)
    ds["strain"] = strain(ds, ref=tuple(cfg.get("strain_ref_ps", (-20, -2))))
    drift = [f"{l['scans'][0]}-{l['scans'][-1]}: q0={l['q0']:.4f}" for l in loops]
    ds.attrs.update({"loops": drift, "q_ref": float(q_ref), "q_window": float(loops[0]["q_window"]),
                     "dropped_frames": int(sum(l["dropped"] for l in loops)),
                     "drift_permille": float((np.nanmax(q0s) - np.nanmin(q0s)) / q_ref * 1e3)
                     if np.isfinite(q0s).sum() > 1 else 0.0})
    return ds


def analyze_series(proposal, series, cfg, inner="pump", qz_center=None, **kw):
    '''
    Runs a series dict {(outer, x): {inner_value: [scans]}} with analyze_curve
    (options: qx_qy_zero, drift_correction, min_frames, delay_tol_ps).
    `inner="pump"` for fluence series (outer = temperature),
    `inner="temperature"` for temperature series (outer = pump).
    Returns an xarray.Dataset with dims (series, <inner>, delay[, q_rel]).
    '''
    import xarray as xr
    outer = "temperature" if inner == "pump" else "pump"

    class _Fmt:          # "80K", "T unbekannt" (no temperature recorded), "pump 9.5"
        def __init__(self, kind):
            self.kind = kind

        def format(self, v):
            if self.kind == "temperature":
                return "T unbekannt" if v < 0 else f"{v:.0f}K"
            return f"pump {v:g}"
    fmt = {"temperature": _Fmt("temperature"), "pump": _Fmt("pump")}
    blocks, log = [], []
    for (o, x), by_inner in series.items():
        label = f"{fmt[outer].format(o)} x{x:+.1f}"
        have = proposal.all()
        n_all = sum(len(v) for v in by_inner.values())
        n_missing = sum(str(s) not in have for v in by_inner.values() for s in v)
        if n_missing:
            print(f"{label}: {n_missing} of {n_all} scans not ingested yet (skipped)", flush=True)
        curves = []
        for v, scans in sorted(by_inner.items()):
            ds = analyze_curve(proposal, scans, cfg, qz_center, **kw)
            if ds is None:
                continue
            log.append({"series": label, inner: v, "loops": len(ds.attrs["loops"]),
                        "drift ‰": round(ds.attrs["drift_permille"], 2), "dropped frames": ds.attrs["dropped_frames"],
                        "max |strain| ‰": round(float(np.abs(ds.strain).max()), 2)})
            print(f"{label}, {fmt[inner].format(v)}: {len(ds.attrs['loops'])} loops, drift between loops "
                  f"{ds.attrs['drift_permille']:.2f} permille, {ds.attrs['dropped_frames']} frames with delay "
                  f"outliers dropped, max |strain| {float(np.abs(ds.strain).max()):.2f} permille", flush=True)
            curves.append(ds.expand_dims({inner: [v]}))
        if not curves:
            continue
        blk = xr.concat(curves, dim=inner, join="outer", combine_attrs="drop")
        blk = blk.expand_dims(series=[label])
        blk = blk.assign_coords({outer: ("series", [o]), "position": ("series", [x])})
        blocks.append(blk)
    if not blocks:
        print("Noch keine Serie mit umgewandelten Scans -> Ingest abwarten (ingest_status) "
              "und erneut ausführen.")
        return None
    out = xr.concat(blocks, dim="series", join="outer", combine_attrs="drop")
    out.attrs = {"sample": proposal.key, "inner": inner,
                 "strain_ref_ps": list(cfg.get("strain_ref_ps", (-20, -2))),
                 "options": json.dumps(kw), "log": json.dumps(log, ensure_ascii=False)}
    return out


def analyze_fluence_series(proposal, series, cfg, qz_center=None, **kw):
    return analyze_series(proposal, series, cfg, inner="pump", qz_center=qz_center, **kw)


def analyze_temperature_series(proposal, series, cfg, qz_center=None, **kw):
    return analyze_series(proposal, series, cfg, inner="temperature", qz_center=qz_center, **kw)


def analyze_rsm(run, smooth=1.0):
    '''
    Cooks a static RSM and locates the main peak (after light smoothing).
    Returns (cooked, {"qx":..., "qy":..., "qz":...}).
    '''
    from scipy.ndimage import gaussian_filter
    r = run.cooked
    im = r["images"].fillna(0)
    sm = gaussian_filter(im.values, smooth)
    i = np.unravel_index(int(np.argmax(sm)), sm.shape)
    return r, {d: float(im[d][k]) for d, k in zip(im.dims, i)}


def suggest_theta_offset(run):
    '''
    Proposes a `theta` offset that puts the RSM main peak at qx = 0
    (symmetric reflection). Re-cooks the RSM with both signs of the
    correction and keeps the better one. Only a suggestion -- apply it with
    spice.anchor/update if it is physically justified.
    '''
    import copy
    spice = {k: v.data() for k, v in run.spice.data.items()}
    recipe = run._load_recipe(spice["recipes"][run.type])
    raw = run.raw.compute()

    def peak(dth):
        sp = copy.deepcopy(spice)
        sp["offsets"]["theta"] = sp["offsets"].get("theta", 0.0) + dth
        im = recipe(run.type, raw, **sp)["images"].fillna(0)
        from scipy.ndimage import gaussian_filter
        sm = gaussian_filter(im.values, 1.0)
        i = np.unravel_index(int(np.argmax(sm)), sm.shape)
        return {d: float(im[d][k]) for d, k in zip(im.dims, i)}

    p0 = peak(0.0)
    d = float(np.degrees(np.arctan2(p0["qx"], p0["qz"])))
    trials = {dth: peak(dth) for dth in (-d, d)}
    best = min(trials, key=lambda k: abs(trials[k]["qx"]))
    cur = spice["offsets"].get("theta", 0.0)
    return {"theta_offset": cur, "peak": p0, "suggested_theta_offset": cur + best,
            "peak_after": trials[best]}


# ---------------------------------------------------------------------------
# plots

def plot_rsm(cooked, peak=None, ax=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    im = cooked["images"].fillna(0)
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, (x, s) in zip(ax, (("qx", "qy"), ("qy", "qx"))):
        m = im.sum(s)
        m.where(m > 0).plot(x=x, y="qz", ax=a, norm=LogNorm(), cmap="viridis")
        if peak:
            a.plot(peak[x], peak["qz"], "r+", ms=12)
    return ax


def plot_series(ds, var="strain", linthresh=10):
    '''One panel per series, one curve per pump (or temperature) value.'''
    import matplotlib.pyplot as plt
    inner = ds.attrs.get("inner", "pump")
    n = ds.sizes["series"]
    fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 3.6), sharey=True, squeeze=False)
    for ax, s in zip(axs[0], ds.series.values):
        d = ds[var].sel(series=s)
        vals = [v for v in d[inner].values if np.isfinite(d.sel({inner: v})).any()]
        for i, v in enumerate(vals):
            c = d.sel({inner: v}).dropna("delay")
            ax.plot(c.delay, c, ".-", ms=3, lw=0.8,
                    color=plt.cm.viridis(i / max(len(vals) - 1, 1)), label=f"{v:g}")
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set(title=str(s), xlabel="delay (ps)")
        ax.legend(title=inner, fontsize=7)
    axs[0, 0].set_ylabel({"strain": "strain (‰)"}.get(var, var))
    fig.suptitle(ds.attrs.get("sample", ""))
    fig.tight_layout()
    return fig


def plot_series_summary(ds, delay_window=(20, 100), var="strain"):
    '''Mean of `var` in a delay window vs. pump (or temperature), one line per series.'''
    import matplotlib.pyplot as plt
    inner = ds.attrs.get("inner", "pump")
    m = ds[var].sel(delay=slice(*delay_window)).mean("delay")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for s in m.series.values:
        c = m.sel(series=s).dropna(inner)
        ax.plot(c[inner], c, "o-", label=str(s))
    ax.set(xlabel=inner, ylabel=f"{var}, delay {delay_window[0]}-{delay_window[1]} ps",
           title=ds.attrs.get("sample", ""))
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig

def peak_window(profile, center=None, halfwidth=0.04):
    '''qz window around the maximum of a 1D profile (near `center` if given).'''
    p = profile if center is None else profile.sel(qz=slice(center - 3 * halfwidth, center + 3 * halfwidth))
    q0 = float(p.qz[int(np.nanargmax(p.values))])
    return (q0 - halfwidth, q0 + halfwidth)


# ---------------------------------------------------------------------------
# CLI

def main():
    import argparse
    ap = argparse.ArgumentParser(description="FemtoMAX -> nx5d")
    ap.add_argument("--config", default="femtomax_config.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("catalog", help="(re)build the scan catalog")
    ing = sub.add_parser("ingest", help="convert scans")
    ing.add_argument("scans", nargs="*", type=int)
    ing.add_argument("--sample")
    ing.add_argument("--type")
    ing.add_argument("--series", choices=["fluence", "temperature", "all"],
                     help="with --sample: RSM + scans of these series only")
    ing.add_argument("--jobs", type=int, default=1, help="parallel processes")
    ing.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(a.config)
    if a.cmd == "catalog":
        df = catalog(cfg, refresh=True)
        print(df.groupby(["sample", "type"], dropna=False).size())
    elif a.cmd == "ingest":
        scans = a.scans
        if not scans:
            df = catalog(cfg)
            if a.series:
                if not a.sample:
                    ap.error("--series needs --sample")
                scans = ingest_order(df, a.sample, a.series)
            else:
                if a.sample:
                    df = df[df["sample"] == a.sample]
                if a.type:
                    df = df[df["type"] == a.type]
                scans = list(df.index)
        print(f"{len(scans)} scans")
        ingest_many(scans, cfg, overwrite=a.overwrite, jobs=a.jobs)


_main = main

if __name__ == "__main__":
    main()
