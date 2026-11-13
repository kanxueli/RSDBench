"""CataNet frame-pool dataset."""
from __future__ import annotations

import fcntl
import importlib.util
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]


def video_complete(pool: Path, vid) -> bool:
    d = Path(pool) / str(vid)
    lab = d / f"{vid}.csv"
    if not lab.is_file():
        return False
    n_rows = sum(1 for _ in open(lab)) - 1
    return n_rows > 0 and n_rows == len(list(d.glob("*.png")))


def _cataract_builder():
    spec = importlib.util.spec_from_file_location(
        "prepare_catanet_frames", _REPO / "scripts" / "cataract101" / "prepare_catanet_frames.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_cataract101(data_root: Path, pool: Path, vid: int, sample_fps: float) -> str:
    mod = _cataract_builder()
    meta = pd.read_csv(data_root / "videos.csv", sep=";")
    ann = pd.read_csv(data_root / "annotations.csv", sep=";")
    row = meta[meta.VideoID == int(vid)].iloc[0]
    vpath = next(p for p in (data_root / "videos" / f"case_{vid}.mp4", data_root / "videos" / f"{vid}.mp4") if p.exists())
    mod.build_one(
        vid=int(vid),
        video_path=vpath,
        ann=ann,
        n_frames=int(row.Frames),
        source_fps=float(row.FPS),
        experience=int(row.Experience),
        out_root=pool,
        sample_fps=sample_fps,
        force=True,
    )
    return f"cataract101 {vid}"


def _build_cholec80(data_root: Path, pool: Path, vid: str, sample_fps: float, dim: int = 256) -> str:
    from data_provider.datasets.cholec80 import PHASE_MAP

    df = pd.read_csv(data_root / "phase_annotations" / f"{vid}-phase.txt", sep="\t")
    phase_src = np.asarray([PHASE_MAP.get(str(p).replace(" ", ""), 0) for p in df.iloc[:, 1]], dtype=np.int64)
    n_src = len(phase_src)
    vpath = data_root / "videos" / f"{vid}.mp4"
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise FileNotFoundError(vpath)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    step = max(1, int(round(source_fps / sample_fps)))
    idxs = np.arange(0, n_src, step, dtype=np.int64)

    out = Path(pool) / vid
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.png"):
        p.unlink()
    want = {int(i): k for k, i in enumerate(idxs.tolist())}
    written = 0
    fidx = 0
    last = int(idxs[-1])
    while fidx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        k = want.get(fidx)
        if k is not None:
            frame = cv2.resize(frame, (dim, dim), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"{vid}_{k + 1:04d}.png"), frame)
            written = k + 1
        fidx += 1
    cap.release()
    if written == 0:
        raise RuntimeError(f"no frames decoded from {vpath}")
    m = written
    pd.DataFrame(
        {
            "step": phase_src[idxs[:m]],
            "u1": np.zeros(m, dtype=np.int64),
            "experience": np.ones(m, dtype=np.int64),
            "u2": np.zeros(m, dtype=np.int64),
            "u3": np.zeros(m, dtype=np.int64),
            "rsd_frames": (n_src - 1 - idxs[:m]).astype(np.int64),
        }
    ).to_csv(out / f"{vid}.csv", index=True)
    return f"cholec80 {vid} frames={m}"


def _build_one(args) -> str:
    dataset, data_root, pool, vid, sample_fps = args
    if dataset == "cataract101":
        return _build_cataract101(Path(data_root), Path(pool), int(vid), sample_fps)
    if dataset == "cholec80":
        return _build_cholec80(Path(data_root), Path(pool), str(vid), sample_fps)
    raise ValueError(f"no CataNet pool builder for dataset={dataset!r}")


def ensure_pool(
    dataset: str,
    data_root: Path,
    pool: Path,
    video_ids: Iterable,
    sample_fps: float = 2.5,
    workers: int = 4,
) -> List[str]:
    """Build every requested video that is missing or incomplete; return the built ids."""
    pool = Path(pool)
    pool.mkdir(parents=True, exist_ok=True)
    ids = [str(v) for v in video_ids]
    with open(pool / ".build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # serialize builders (CataNet + BDNet may start together)
        missing = [v for v in ids if not video_complete(pool, v)]
        if not missing:
            return []
        print(f"[pool] {dataset}: building {len(missing)} video(s) into {pool}", flush=True)
        jobs = [(dataset, str(data_root), str(pool), v, float(sample_fps)) for v in missing]
        if workers <= 1 or len(jobs) == 1:
            for j in jobs:
                print(f"[pool] {_build_one(j)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for msg in ex.map(_build_one, jobs):
                    print(f"[pool] {msg}", flush=True)
        bad = [v for v in missing if not video_complete(pool, v)]
        if bad:
            raise RuntimeError(f"[pool] still incomplete after build: {bad}")
        return missing
