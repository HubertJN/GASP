import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers 

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

field_colors = {}

icol = 0

# Read the Excel file
df = pd.read_excel('data/2DIsing_rates.xlsx')

brendel_data = []
brendel_marker = "^"

# Grab the entries from the Brendel at al paper
brendel = df[df['source'] == "Brendel et al"]
num_fields = brendel['h'].nunique()
brendel_fields = brendel['h'].unique()

for field in brendel_fields:

    if (field not in field_colors):
        field_colors[field] = colors[icol]
        icol += 1

    brendel_beta = brendel[brendel['h']==field]['beta'].to_numpy()
    brendel_rate = brendel[brendel['h']==field]['I (MCSS^1)'].to_numpy()
    brendel_data.append((field, brendel_beta, brendel_rate))

# Grab the entries computed with GASP
gasp = df[df['source'] == "GPU_2DIsing"]
num_fields = gasp['h'].nunique()
gasp_fields = gasp['h'].round(2).unique()

gasp_data = []
gasp_marker = "o"

for field in gasp_fields:

    if (field not in field_colors):
        field_colors[field] = colors[icol%10]
        icol += 1

    gasp_beta = gasp[gasp['h'].round(2)==field]['beta'].to_numpy()
    gasp_rate = gasp[gasp['h'].round(2)==field]['I (MCSS^1)'].to_numpy()
    gasp_data.append((field, gasp_beta, gasp_rate))

ryu = df[df['source'] == "Ryu and Cai"]
num_fields = ryu['h'].nunique()
ryu_fields = ryu['h'].round(3).unique()

ryu_data = []
ryu_marker = 'X'

for field in ryu_fields:

    if (field not in field_colors):
        field_colors[field] = colors[icol%10]
        icol += 1
    
    ryu_beta = ryu[ryu['h'].round(3)==field]['beta'].to_numpy()
    ryu_rate = ryu[ryu['h'].round(3)==field]['I (MCSS^1)'].to_numpy()
    ryu_data.append((field, ryu_beta, ryu_rate))

# Compute horizontal line value
y_hline = gasp_data[3][2][2]
plt.hlines(y_hline, xmin=0.48, xmax=0.85)

# Check intersections
def find_intersections(beta, rate, y_val):
    intersections = []
    for i in range(len(beta) - 1):
        y1, y2 = rate[i], rate[i+1]
        if (y1 - y_val) * (y2 - y_val) < 0:  # sign change => crossing
            # linear interpolation
            x1, x2 = beta[i], beta[i+1]
            x_cross = x1 + (y_val - y1) * (x2 - x1) / (y2 - y1)
            intersections.append(x_cross)
    return intersections

# Plot data and compute intersections
for i, data in enumerate(ryu_data):
    if data[0] >= 0.04 and data[0] <= 0.11:
        plt.plot(data[1], data[2], marker=ryu_marker, color=field_colors[data[0]], linestyle='')
        xs = find_intersections(data[1], data[2], y_hline)
        for x in xs:
            print(f"Ryu field {data[0]} intersects at x = {x:.4f}")

for i, data in enumerate(gasp_data):
    plt.plot(data[1], data[2], marker=gasp_marker, label=f'h={data[0]}', linestyle='--', color=field_colors[data[0]])
    xs = find_intersections(data[1], data[2], y_hline)
    for x in xs:
        print(f"GASP field {data[0]} intersects at x = {x:.4f}")

for i, data in enumerate(brendel_data):
    plt.plot(data[1], data[2], marker=brendel_marker, color=field_colors[data[0]], linestyle='--')
    xs = find_intersections(data[1], data[2], y_hline)
    for x in xs:
        print(f"Brendel field {data[0]} intersects at x = {x:.4f}")

plt.yscale('log')
plt.xlabel(r"$\beta$")
plt.ylabel(r"I MCSS$^{-1}$")
plt.xlim((0.48, 0.85))
plt.ylim((1e-15, 1e-5))
plt.legend()
plt.savefig('figures/rates.pdf')

plt.yscale('log')
plt.xlabel(r"$\beta$")
plt.ylabel(r"I MCSS$^{-1}$")
plt.xlim((0.48,0.7))
plt.ylim((1e-15,1e-5))
plt.legend()
plt.savefig('figures/rates.pdf')
