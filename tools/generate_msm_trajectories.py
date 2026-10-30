"""Generate short MSM trajectories from training configurations and store them in HDF5.

Each GASP call takes one seed configuration per GPU multiprocessor and runs
`replicas` independent copies of each for `max(lags)` sweeps, keeping the frames
at every lag. The run never resolves early: the thresholds are set outside the
reachable magnetisation range, so every replica runs the full length.

Usage
-----
    python -m tools.generate_msm_trajectories --beta 0.550 --h 0.070

File layout (data/trajectories/trajectories_{beta:.3f}_{h:.3f}.hdf5)
----------------------------------------------------------------------
attrs                 L, beta, h, replicas, rng_seed, source, gasp_commit, ...
lags                  (n_lags,) int     sweeps at which frames were kept
seed_index            (n_seeds,) int    row of each seed in the training file
seed_attrs            (n_seeds, 4)      training attrs row: magnetisation, LGCS, p_B, p_B_err
seed_grids            (n_seeds, L*L/8)  uint8, seed configuration as packed bits
frames                (n_seeds, replicas, n_lags, L*L/8) uint8, packed bits
run_seed              (n_runs,) uint64  RNG seed passed to GASP for each run
run_done              (n_runs,) bool    set once a run's frames are written
basin_frames          (nsms, replicas, n_lags, L*L/8) uint8, the same run started
                      from all-down grids: a reference sample of the parent basin
basin_seed            () uint64         RNG seed of the basin run

Grids are packed exactly as in the training files: spin +1 -> bit 1, flattened
row-major, np.packbits(..., bitorder="little"). Run k covers seeds
k*nsms ... (k+1)*nsms - 1. Rerunning the command resumes at the first run
whose run_done is False.
"""

import argparse
import datetime
import subprocess
import time
from pathlib import Path

import gasp
import h5py
import numpy as np

from tools.utils import load_into_array

DEFAULT_LAGS = [8 * 2**k for k in range(11)]  # 8 ... 8192


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--h", type=float, required=True)
    p.add_argument("--runs", type=int, default=50, help="GASP calls; seeds = runs * nsms")
    p.add_argument("--replicas", type=int, default=128, help="replicas per seed")
    p.add_argument("--lags", type=int, nargs="+", default=DEFAULT_LAGS)
    p.add_argument("--rng-seed", type=int, default=0, help="seeds both seed selection and GASP")
    p.add_argument("--training", default=None, help="default data/gridstates_training_{beta}_{h}.hdf5")
    p.add_argument("--out", default=None, help="default data/trajectories/trajectories_{beta}_{h}.hdf5")
    return p.parse_args()


def pack(grids):
    """Pack (n, L, L) +-1 grids into (n, L*L/8) uint8, matching the training files."""
    n = grids.shape[0]
    return np.packbits((grids > 0).reshape(n, -1), axis=1, bitorder="little")


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def create_file(path, args, training_path, L, nsms):
    """Create the output file, choosing seeds and per-run RNG seeds up front."""
    _, attrs, _ = load_into_array(training_path, load_grids=False)
    n_seeds = args.runs * nsms
    if n_seeds > len(attrs):
        raise ValueError(f"{n_seeds} seeds requested but only {len(attrs)} training grids")

    rng = np.random.default_rng(args.rng_seed)
    seed_index = np.sort(rng.choice(len(attrs), size=n_seeds, replace=False))
    run_seed = rng.integers(1, 2**63, size=args.runs + 1, dtype=np.uint64)
    seed_grids, _, _ = load_into_array(training_path, indices=seed_index)

    nbytes = L * L // 8
    lags = np.asarray(sorted(args.lags), dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs.update({
            "L": L, "beta": args.beta, "h": args.h, "replicas": args.replicas, "nsms": nsms,
            "rng_seed": args.rng_seed, "source": str(training_path), "gasp_commit": git_commit(),
            "gpu_method": 2, "threads_per_block": 32,
            "created": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec="seconds"),
        })
        f["lags"] = lags
        f["seed_index"] = seed_index
        f["seed_attrs"] = attrs[seed_index]
        f["seed_grids"] = pack(seed_grids)
        f["run_seed"] = run_seed[:-1]
        f["run_done"] = np.zeros(args.runs, dtype=bool)
        f["basin_seed"] = run_seed[-1]
        f.create_dataset(
            "frames", shape=(n_seeds, args.replicas, len(lags), nbytes), dtype=np.uint8,
            chunks=(1, args.replicas, len(lags), nbytes),
        )


def run_batch(seeds, replicas, lags, beta, h, seed):
    """Run `replicas` copies of each seed grid and return frames (n_seeds, replicas, n_lags, L, L)."""
    nsms = len(seeds)
    L = seeds[0].shape[0]
    ngrids = nsms * replicas
    gasp.run_committor_calc(
        L, ngrids, int(lags[-1]) + 1, beta, h,
        grid_output_int=int(lags[0]), mag_output_int=int(lags[0]),
        grid_output_sweeps=[int(m) for m in lags], seed=int(seed),
        grid_input="NumPy", grid_array=list(seeds),
        keep_grids=True, up_threshold=1.01, dn_threshold=-1.01,
        nsms=nsms, gpu_method=2, outname="None",
    )
    swept = [snap[0].isweep for snap in gasp.grids]
    if swept != list(lags):
        raise RuntimeError(f"GASP returned frames at sweeps {swept}, expected {list(lags)}")

    # gasp.grids[k][i] is replica i at lag k; replica i belongs to seed i // replicas
    frames = np.empty((nsms, replicas, len(lags), L, L), dtype=np.int8)
    for k, snap in enumerate(gasp.grids):
        frames[:, :, k] = np.stack([g.grid for g in snap]).reshape(nsms, replicas, L, L)
    return frames


def main():
    args = parse_args()
    tag = f"{args.beta:.3f}_{args.h:.3f}"
    training_path = Path(args.training or f"data/gridstates_training_{tag}.hdf5")
    out = Path(args.out or f"data/trajectories/trajectories_{tag}.hdf5")
    nsms = gasp.gpu_nsms

    with h5py.File(training_path, "r") as f:
        L = int(f["L"][()])
    if not out.exists():
        create_file(out, args, training_path, L, nsms)

    with h5py.File(out, "r") as f:
        if f.attrs["nsms"] != nsms or f.attrs["replicas"] != args.replicas:
            raise ValueError(f"{out} was made with different nsms/replicas; delete it to start over")
        lags = f["lags"][()]
        run_seed = f["run_seed"][()]
        run_done = f["run_done"][()]
        packed_seeds = f["seed_grids"][()]
        basin_seed = f["basin_seed"][()]
        basin_done = "basin_frames" in f

    seeds = np.unpackbits(packed_seeds, axis=1, bitorder="little")[:, : L * L]
    seeds = (seeds.astype(np.int8) * 2 - 1).reshape(-1, L, L)

    t0 = time.perf_counter()
    if not basin_done:
        down = -np.ones((nsms, L, L), dtype=np.int8)
        frames = run_batch(down, args.replicas, lags, args.beta, args.h, basin_seed)
        packed = pack(frames.reshape(-1, L, L)).reshape(nsms, args.replicas, len(lags), -1)
        with h5py.File(out, "a") as f:
            f["basin_frames"] = packed
        print(f"[{tag}] basin reference run done", flush=True)

    for k in np.flatnonzero(~run_done):
        block = slice(k * nsms, (k + 1) * nsms)
        frames = run_batch(seeds[block], args.replicas, lags, args.beta, args.h, run_seed[k])
        packed = pack(frames.reshape(-1, L, L)).reshape(nsms, args.replicas, len(lags), -1)
        with h5py.File(out, "a") as f:
            f["frames"][block] = packed
            f["run_done"][k] = True
        print(f"[{tag}] run {k + 1}/{len(run_done)} done, {time.perf_counter() - t0:.0f} s elapsed",
              flush=True)

    print(f"[{tag}] complete: {out}")


if __name__ == "__main__":
    main()
