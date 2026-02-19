import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import csv
import os
from tools.utils import load_config, load_into_array

# --- parse optional inputs ---
parser = argparse.ArgumentParser()
parser.add_argument("--h", type=float, help="External field value")
parser.add_argument("--beta", type=float, help="Inverse temperature")
args = parser.parse_args()

# --- Load config ---
config = load_config("config.yaml")

# --- apply overrides if provided ---
h = args.h if args.h is not None else config.parameters.h
beta = args.beta if args.beta is not None else config.parameters.beta

h5path = f"data/gridstates_{beta:.3f}_{h:.3f}.hdf5"
plot_dir = config.paths.plot_dir

print("Opening:", h5path)

_, attrs, _ = load_into_array(h5path, load_grids=False)

# --- Prune invalid entries ---
mask = (attrs[:, 0] != -1) & (attrs[:, 1] != 0)
attrs = attrs[mask]

print(f"Processed {attrs.shape[0]} attributes.")

magnetizations = attrs[:, 0]
cluster = attrs[:, 1]

# --- Magnetization histogram ---
plt.figure(figsize=(12, 6))
plt.hist(magnetizations, bins=config.analyse.bins, color='skyblue', edgecolor='black')
plt.xlabel('Magnetization')
plt.ylabel('Frequency')
plt.title('Histogram of Magnetization Values')
plt.xlim(-1, 1)
xmin, xmax = plt.xlim()
plt.xticks(np.arange(round(xmin, 1), round(xmax + 0.1, 1), 0.1))
plt.tight_layout()
plt.savefig(f"{plot_dir}/magnetization_hist_{beta:.3f}_{h:.3f}.pdf")
plt.close()

# --- Cluster histogram ---
cluster_min = config.analyse.cluster_min
cluster_max = config.analyse.cluster_max
bins = np.arange(cluster_min - 0.5, cluster_max + 1.5, 1) if config.analyse.bin_per_cluster else config.analyse.bins

plt.figure(figsize=(8, 6))
counts, bins, _ = plt.hist(cluster, bins=bins, color='skyblue', edgecolor='black')

max_idx = np.argmax(counts)
bin_center = (bins[max_idx] + bins[max_idx + 1]) / 2

plt.text(bin_center, counts[max_idx] + 1, f'Bin: {int(bin_center)}',
         ha='center', va='bottom', color='red', fontsize=12)
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.title('Histogram of Cluster Values')
plt.xlim(cluster_min - 0.5, 100.5)

plt.tight_layout()
plt.savefig(f"{plot_dir}/cluster_hist_{beta:.3f}_{h:.3f}.pdf")
plt.close()

dn = int(bin_center)
print(dn)

csv_path = "h_beta_dn.csv"

# Append to CSV (write header only if the file doesn't exist or is empty)
needs_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if needs_header:
        writer.writerow(["h", "beta", "dn"])
    writer.writerow([f"{h:.3f}", f"{beta:.3f}", int(dn)])