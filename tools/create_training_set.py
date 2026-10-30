import argparse
import math

import gasp
import h5py
import matplotlib.pyplot as plt
import numpy as np

from tools.utils import load_config, load_into_array, save_training_grids


def allocate_with_caps(weights: np.ndarray, caps: np.ndarray, total: int) -> np.ndarray:
    """
    Allocate `total` integer samples across bins in proportion to `weights`,
    while respecting per-bin capacities `caps`.
    """
    weights = np.asarray(weights, dtype=float)
    caps = np.asarray(caps, dtype=int)

    if total <= 0 or caps.sum() == 0:
        return np.zeros_like(caps, dtype=int)

    weights = np.maximum(weights, 0.0)
    active = caps > 0
    if not np.any(active):
        return np.zeros_like(caps, dtype=int)

    if weights[active].sum() == 0:
        weights = active.astype(float)

    alloc = np.zeros_like(caps, dtype=int)
    remaining = min(total, int(caps.sum()))
    available = caps.copy()

    while remaining > 0:
        active = available > 0
        if not np.any(active):
            break

        active_weights = weights.copy()
        active_weights[~active] = 0.0
        wsum = active_weights.sum()
        if wsum == 0:
            active_weights = active.astype(float)
            wsum = active_weights.sum()

        target = active_weights / wsum * remaining
        add = np.floor(target).astype(int)
        add = np.minimum(add, available)

        added_now = int(add.sum())
        alloc += add
        available -= add
        remaining -= added_now

        if remaining == 0:
            break

        # Distribute leftovers by largest fractional parts among bins with capacity
        frac = target - np.floor(target)
        frac[available <= 0] = -1.0
        order = np.argsort(-frac)

        progressed = False
        for i in order:
            if remaining == 0:
                break
            if available[i] > 0:
                alloc[i] += 1
                available[i] -= 1
                remaining -= 1
                progressed = True

        if not progressed:
            break

    return alloc


# --- parse optional beta and h inputs ---
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Override beta value")
parser.add_argument("--h", type=float, help="Override h value")
parser.add_argument("--target", type=float, help="Target value for sampling")
parser.add_argument("--seed", type=int, default=42, help="RNG seed")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# --- Load config ---
config = load_config("config.yaml")

beta = args.beta if args.beta is not None else config.parameters.beta
h = args.h if args.h is not None else config.parameters.h
target_val = args.target if args.target is not None else config.collective_variable.target

# --- Load only headers and attributes first ---
inname = config.paths.gridstates
outname = config.paths.training
if args.beta is not None or args.h is not None:
    inname = f"data/gridstates_{beta:.3f}_{h:.3f}.hdf5"
    outname = f"data/gridstates_training_{beta:.3f}_{h:.3f}.hdf5"

_, attrs, headers = load_into_array(inname, load_grids=False)

val_min = 1
val_max = config.collective_variable.up_threshold
num_bins = 256
max_samples = 140 * 8 * config.gpu.tasks

gpu_nsms = gasp.gpu_nsms - gasp.gpu_nsms % config.gpu.sm_mult

values = attrs[:, 1]
mask = (values >= val_min) & (values <= val_max)
candidates = np.where(mask)[0]
cand_val = values[mask]
max_samples = min(max_samples, len(candidates))

# More interpretable target shaping
sigma = max(abs(0.33 * target_val), 0.05 * (val_max - val_min))
alpha_uniform = 0.15  # 15% uniform background, 85% target-focused
skew = 0.0

if len(candidates) == 0:
    sample_idx = np.array([], dtype=int)
    bins = np.linspace(val_min, val_max, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    desired_mass = np.ones(num_bins) / num_bins
else:
    bins = np.linspace(val_min, val_max, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_counts, _ = np.histogram(cand_val, bins=bins)

    z = (bin_centers - target_val) / sigma
    focused_pdf = np.exp(-0.5 * z**2) * (1 + np.tanh(skew * z))
    focused_pdf = np.maximum(focused_pdf, 0.0)

    uniform_mass = np.ones(num_bins) / num_bins
    if focused_pdf.sum() == 0:
        desired_mass = uniform_mass
    else:
        focused_mass = focused_pdf / focused_pdf.sum()
        desired_mass = (1 - alpha_uniform) * focused_mass + alpha_uniform * uniform_mass

    cap = min(max_samples, len(candidates))
    gpu_multiple = gpu_nsms
    target_samples = min(
        len(candidates) - (len(candidates) % gpu_multiple),
        cap - (cap % gpu_multiple),
        len(candidates),
    )

    desired_counts = allocate_with_caps(desired_mass, bin_counts, target_samples)

    chosen = []
    actual_per_bin = []

    for i in range(num_bins):
        k = int(desired_counts[i])
        if i < num_bins - 1:
            bin_mask = (cand_val >= bins[i]) & (cand_val < bins[i + 1])
        else:
            bin_mask = (cand_val >= bins[i]) & (cand_val <= bins[i + 1])

        idxs_in_bin = candidates[bin_mask]
        k = min(k, len(idxs_in_bin))
        actual_per_bin.append(k)

        if k > 0:
            sel = rng.choice(idxs_in_bin, size=k, replace=False)
            chosen.append(sel)

    sample_idx = np.hstack(chosen) if chosen else np.array([], dtype=int)

    # Compute LCM of gpu_nsms and tasks
    divisor = abs(gpu_nsms * config.gpu.tasks) // math.gcd(gpu_nsms, config.gpu.tasks)

    # Trim randomly to preserve distribution, not by linspace
    n = len(sample_idx)
    n_trim = n - (n % divisor)
    if 0 < n_trim < n:
        sample_idx = rng.choice(sample_idx, size=n_trim, replace=False)

    print(
        f"Requested target_samples={target_samples}, selected_pretrim={n}, "
        f"selected_posttrim={len(sample_idx)}, divisor={divisor}"
    )
    print(
        f"Bin allocation summary: occupied_bins={(bin_counts > 0).sum()}, "
        f"nonzero_alloc={(desired_counts > 0).sum()}, max_bin_count={bin_counts.max()}"
    )

if sample_idx.size > 0:
    sample_idx = np.sort(sample_idx)
    sample_grids, _, _ = load_into_array(inname, load_grids=True, indices=sample_idx)
    sample_attrs = attrs[sample_idx]
    order = np.argsort(sample_attrs[:, 1])
    sample_idx = sample_idx[order]
    sample_attrs = sample_attrs[order]
    sample_grids = sample_grids[order]
else:
    sample_grids = []
    sample_attrs = np.empty((0, attrs.shape[1]))

print(
    "Selected",
    len(sample_grids),
    "Percentage of total",
    int(len(sample_grids) / len(attrs) * 100) if len(attrs) else 0,
    "%",
)

plt.figure(figsize=(6, 4))
plt.hist(values[sample_idx], bins=bins, edgecolor="black", rwidth=0.9, alpha=0.6)
expected_counts = desired_mass * len(sample_idx)
plt.plot(bin_centers, expected_counts, "r--", linewidth=2)
plt.xlabel("Values")
plt.ylabel("Count")
plt.savefig(f"figures/training_samples_{beta:.3f}_{h:.3f}.pdf")
plt.close()

save_training_grids(outname, sample_grids, sample_attrs, headers)