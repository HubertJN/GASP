import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tools.utils import load_into_array, save_training_grids, load_config
import gasp
import math

# --- parse optional beta and h inputs ---
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Override beta value")
parser.add_argument("--h", type=float, help="Override h value")
parser.add_argument("--target", type=float, help="Target value for sampling")
args = parser.parse_args()

# --- Load config ---
config = load_config("config.yaml")

# Apply overrides if provided
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

tolerance = abs(0.33*target_val)
sigma = tolerance
size_mult = 0.2
skew = 0

val_min = 1
val_max = config.collective_variable.up_threshold
num_bins = 256
max_samples = 140*8*config.gpu.tasks

gpu_nsms = gasp.gpu_nsms - gasp.gpu_nsms % config.gpu.sm_mult

values = attrs[:, 1]
mask = (values >= val_min) & (values <= val_max)
candidates, cand_val = np.where(mask)[0], values[mask]
max_samples = min(max_samples, len(attrs))

if len(candidates) == 0:
    sample_idx = np.array([], dtype=int)
else:
    bins = np.linspace(val_min, val_max, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_counts, _ = np.histogram(cand_val, bins=bins)
    nonempty = bin_counts > 0
    z = (bin_centers - target_val) / sigma
    desired_pdf = np.exp(-0.5 * z**2) * (1 + np.tanh(skew * z)) * size_mult
    desired_pdf += 1/num_bins
    desired_mass = desired_pdf / desired_pdf.sum()
    total_candidates = len(candidates)
    gpu_multiple, cap = gpu_nsms, max_samples
    target_samples = min(total_candidates - (total_candidates % gpu_multiple),
                        cap - (cap % gpu_multiple),
                        total_candidates)
    desired_counts_float = desired_mass * target_samples
    desired_counts = np.floor(desired_counts_float).astype(int)
    remaining = target_samples - desired_counts.sum()
    if remaining > 0:
        frac = desired_counts_float - desired_counts
        order = np.argsort(-frac)
        for i in order:
            if remaining == 0: break
            if desired_counts[i] < bin_counts[i]:
                desired_counts[i] += 1
                remaining -= 1
    chosen = []
    for i in range(num_bins):
        k = desired_counts[i]
        if k <= 0: continue
        if i < num_bins - 1:
            bin_mask = (cand_val >= bins[i]) & (cand_val < bins[i+1])
        else:
            bin_mask = (cand_val >= bins[i]) & (cand_val <= bins[i+1])
        idxs_in_bin = candidates[bin_mask]
        k = min(k, len(idxs_in_bin))
        if len(idxs_in_bin) == 0: continue
        sel = np.random.choice(idxs_in_bin, size=k, replace=False)
        chosen.append(sel)
    sample_idx = np.hstack(chosen) if chosen else np.array([], dtype=int)

    # Compute LCM of gpu_nsms and SLURM tasks
    divisor = abs(gpu_nsms * config.gpu.tasks) // math.gcd(gpu_nsms, config.gpu.tasks)

    # Trim sample_idx to be multiple of divisor
    n = len(sample_idx)
    n_trim = n - (n % divisor)
    if n_trim > 0:
        selected_idx = np.linspace(0, n-1, n_trim, dtype=int)
        sample_idx = sample_idx[selected_idx]

if sample_idx.size > 0:
    sample_idx = np.sort(sample_idx)
    sample_grids, _, _ = load_into_array(inname, load_grids=True, indices=sample_idx)
    sample_attrs = attrs[sample_idx]
    order = np.argsort(sample_attrs[:, 1])
    sample_idx   = sample_idx[order]
    sample_attrs = sample_attrs[order]
    sample_grids = sample_grids[order]
else:
    sample_grids = []
    sample_attrs = np.empty((0, attrs.shape[1]))

print("Selected", len(sample_grids),
      "Percentage of total", int(len(sample_grids) / len(attrs) * 100), "%")

plt.figure(figsize=(6,4))
plt.hist(values[sample_idx], bins=bins, edgecolor='black', rwidth=0.9, alpha=0.6)
expected_counts = (desired_mass / desired_mass.sum()) * len(sample_idx)
plt.plot(bin_centers, expected_counts, 'r--', linewidth=2)
plt.xlabel("Values")
plt.ylabel("Count")
plt.savefig("figures/training_samples.pdf")
plt.close()

save_training_grids(outname, sample_grids, sample_attrs, headers)
