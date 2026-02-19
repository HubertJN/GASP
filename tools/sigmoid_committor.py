import numpy as np
import h5py
import gasp
import argparse
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import curve_fit
from tools.utils import load_into_array, magnetisation, load_config

# --- parse optional beta and h inputs ---
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Override beta value")
parser.add_argument("--h", type=float, help="Override h value")
parser.add_argument("--dn", type=float, help="Override dn_threshold value")
args = parser.parse_args()

# --- Load config ---
config = load_config("config.yaml")

# Apply overrides if provided
beta = args.beta if args.beta is not None else config.parameters.beta
h = args.h if args.h is not None else config.parameters.h
dn_threshold = args.dn if args.dn is not None else config.collective_variable.dn_threshold
dn_threshold -= 0.5
up_threshold = config.collective_variable.up_threshold

# --- Load only headers and attributes first ---
inname = config.paths.gridstates
if args.beta is not None or args.h is not None:
    inname = f"data/gridstates_{beta:.3f}_{h:.3f}.hdf5"

_, attrs, headers = load_into_array(inname, load_grids=False)

L = headers["L"]
nsweeps = headers["tot_nsweeps"]
beta = headers["beta"]
h = headers["h"]

# --- Prune invalid entries ---
mask = (attrs[:, 0] != -1) & (attrs[:, 1] != 0)
attrs = attrs[mask]

cluster = attrs[:, 1]

gpu_nsms = gasp.gpu_nsms - gasp.gpu_nsms % config.gpu.sm_mult
ngrids = 4 * gpu_nsms * 32
conc_calc = min(int(np.round(ngrids / config.sig.ngrids)), gpu_nsms)

# Targets to sample (your choice; this is what you had)
clust_to_sample = np.linspace(cluster.min(), cluster.max(), conc_calc)

# Nearest-neighbour selection (allows duplicates if cluster is sparse)
sample_idx = np.array([np.argmin(np.abs(cluster - t)) for t in clust_to_sample], dtype=int)

# Sort by index to maintain order in original dataset (important for loading grids later)
sample_idx.sort()

print(f"Sampling {len(sample_idx)} grids from cluster values: {cluster[sample_idx].sort()}")
print(f"Running {conc_calc} concurrent calculation on {int(ngrids/conc_calc)} grids each")

# Load only sampled grids (order corresponds to sample_idx)
sample_grids, _, _ = load_into_array(inname, load_grids=True, indices=sample_idx)

# Preallocate arrays to store results for ALL sampled points
pB_mean = np.empty(len(sample_idx), dtype=float)
pB_err = np.empty(len(sample_idx), dtype=float)

# Iterate over sampled grids in chunks of size conc_calc
# IMPORTANT: sample_grids is indexed 0..len(sample_idx)-1, not by original sample_idx values
for start in range(0, len(sample_idx), conc_calc):
    end = min(start + conc_calc, len(sample_idx))

    gridlist = [g.copy() for g in sample_grids[start:end]]

    pBfast = np.array(
        gasp.run_committor_calc(
            L, ngrids, config.sig.nsweeps, beta, h,
            grid_output_int=50000,
            mag_output_int=1,
            grid_input="NumPy",
            grid_array=gridlist,
            cv=config.collective_variable.type,
            dn_threshold=dn_threshold,
            up_threshold=up_threshold,
            keep_grids=False,
            nsms=gpu_nsms,
            gpu_method=2
        )
    )

    pB_mean[start:end] = pBfast[:, 0]
    pB_err[start:end] = pBfast[:, 1]

# Final array: [cluster, committor, committor_err] for sampled indices
result = np.column_stack([cluster[sample_idx], pB_mean, pB_err])

print("Sampled [cluster, committor, committor_err] (first 10 rows):")
print(result[:10])

# =======================
# --- Sigmoid Fit ---
# =======================
test_cluster_size = result[:, 0]
test_committor = result[:, 1]
test_error = result[:, 2]

def sigmoid(x, k, x0):
    base = 1 / (1 + np.exp(-k * (x - x0)))
    base0 = 1 / (1 + np.exp(k * x0))
    return (base - base0) / (1 - base0 + 1e-10)

try:
    with np.errstate(divide='ignore', invalid='ignore'):
        popt, _ = curve_fit(
            sigmoid,
            test_cluster_size,
            test_committor,
            p0=[0.1, np.median(test_cluster_size)],
            maxfev=10000,
        )
except RuntimeError as e:
    print(f"Sigmoid fit failed: {e}, using default parameters")
    popt = [0.1, np.median(test_cluster_size)]
x_fit = np.linspace(test_cluster_size.min(), test_cluster_size.max(), 200)
y_fit = sigmoid(x_fit, *popt)

# Print cluster that corresponds to committor ~0.5
x0 = popt[1]
print(int(x0))

plt.figure(figsize=(6,6))
plt.errorbar(
    test_cluster_size,
    test_committor,
    yerr=test_error,
    fmt='o',
    ms=3,
    alpha=0.5,
    label="Test Data",
    ecolor='gray',
    capsize=2,
)
plt.plot(x_fit, y_fit, label="Sigmoid Fit", color='black', linewidth=2)
plt.xlabel("Cluster Size")
plt.ylabel("Committor")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{config.paths.plot_dir}/sigmoid_fit_{beta:.3f}_{h:.3f}.png", dpi=300, bbox_inches='tight')

csv_path = "h_beta_dn_target.csv"

# Append to CSV (write header only if the file doesn't exist or is empty)
needs_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if needs_header:
        writer.writerow(["h", "beta", "dn", "target"])
    writer.writerow([f"{h:.3f}", f"{beta:.3f}", int(dn_threshold), int(x0)])
