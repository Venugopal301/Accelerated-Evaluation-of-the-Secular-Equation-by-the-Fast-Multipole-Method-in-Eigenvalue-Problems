#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:05:15 2026

@author: venugopal
"""

import numpy as np
import matplotlib.pyplot as plt

from check_interlacing_sigma_pos import check_interlacing_sigma_pos


#"""
data = np.load("fmm_random_results_r_20.npz", allow_pickle=True)

setup_time    = data["setup_time"]
elapsed_time  = data["elapsed_time"]
fmm_fun_val   = data["fmm_fun_val"]
fmm_y         = data["fmm_y"]
flop_counts   = data["flop_counts"]
exact_fun_val = data["exact_fun_val"]
exact_y       = data["exact_y"]

print("Data loaded successfully.")


#--------------- HISTOGRAM ----------------------

fmm_vals = np.abs(fmm_fun_val[0])
exact_vals = np.abs(exact_fun_val[0])

# ==================================================
# Magnitude bins
# ==================================================

bins = np.concatenate((10.0**np.arange(-13, 0), [np.inf]))

labels = [
    r"$10^{-13}$-$10^{-12}$",
    r"$10^{-12}$-$10^{-11}$",
    r"$10^{-11}$-$10^{-10}$",
    r"$10^{-10}$-$10^{-9}$",
    r"$10^{-9}$-$10^{-8}$",
    r"$10^{-8}$-$10^{-7}$",
    r"$10^{-7}$-$10^{-6}$",
    r"$10^{-6}$-$10^{-5}$",
    r"$10^{-5}$-$10^{-4}$",
    r"$10^{-4}$-$10^{-3}$",
    r"$10^{-3}$-$10^{-2}$",
    r"$10^{-2}$-$10^{-1}$",
    r"$10^{-1}$-$\infty$"
]

# ==================================================
# Histogram counts
# ==================================================

fmm_counts, _ = np.histogram(fmm_vals, bins=bins)
exact_counts, _ = np.histogram(exact_vals, bins=bins)

# Print counts
print("Magnitude Bin".ljust(25), "FMM".rjust(8), "Exact".rjust(8))
print("-"*45)

for lab, fc, ec in zip(labels, fmm_counts, exact_counts):
    print(f"{lab:<25}{fc:>8}{ec:>8}")

# ==================================================
# Grouped bar chart
# ==================================================

x = np.arange(len(labels))
width = 0.38

plt.figure(figsize=(12,6))

bars1 = plt.bar(x - width/2, fmm_counts, width,
                label='FMM')

bars2 = plt.bar(x + width/2, exact_counts, width,
                label='Exact')

# Counts above bars
for bar in bars1:
    h = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2,
             h,
             f'{int(h)}',
             ha='center',
             va='bottom',
             fontsize=8)

for bar in bars2:
    h = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2,
             h,
             f'{int(h)}',
             ha='center',
             va='bottom',
             fontsize=8)

plt.xticks(x, labels, rotation=45, ha='right')
plt.xlabel(r"Residual Magnitude $|f(y)|$")
plt.ylabel("Number of Roots")
plt.title(r"Residual Distribution for $n=50000,\; r=20$")
plt.legend()

plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


#-------------------------TIMGING COMPARISON --------------------

# Newton iteration numbers
iters = np.arange(1, len(elapsed_time[0][0]) + 1)

# Problem sizes
sizes = [5000, 10000, 15000]
indices = [0, 1, 2]

plt.figure(figsize=(8,6))

for idx, n in zip(indices, sizes):
    plt.plot(iters, elapsed_time[idx][0],
             marker='o',
             linewidth=2,
             label=f'FMM  (n={n})')

    plt.plot(iters, elapsed_time[idx][1],
             marker='s',
             linewidth=2,
             linestyle='--',
             label=f'Exact (n={n})')

plt.xlabel("Newton Iteration")
plt.ylabel("Time (seconds)")
plt.title("Timing Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Newton iteration numbers
iters = np.arange(1, len(elapsed_time[8][0]) + 1)

sizes = [45000, 50000]
indices = [8, 9]

plt.figure(figsize=(8,6))

for idx, n in zip(indices, sizes):
    plt.plot(iters, elapsed_time[idx][0],
             marker='o',
             linewidth=2,
             label=f'FMM  (n={n})')

    plt.plot(iters, elapsed_time[idx][1],
             marker='s',
             linewidth=2,
             linestyle='--',
             label=f'Exact (n={n})')

plt.xlabel("Newton Iteration")
plt.ylabel("Time (seconds)")
plt.title("Timing Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------- RELATIVE ERRORS ------------------------

# Relative error of FMM roots with respect to exact roots
data = np.load("fmm_random_results_r_60.npz", allow_pickle=True)
fmm_y         = data["fmm_y"]

rel_err = np.zeros(10)
for jj in range(10):
    rel_err[jj] = np.linalg.norm(fmm_y[jj] - exact_y[jj]) / np.linalg.norm(exact_y[jj])
    
print("Relative errors:")
print(rel_err)

np.savez(
    "relative_errors_r_60.npz",
    rel_err=rel_err
)

print("Relative errors saved successfully.")


# ---------- PLOT OF RELATIVE ERRORS ALL IN ONE PLOT--------------

# ==========================================
# Ranks to compare
# ==========================================

r_values = [20, 30, 40, 50, 60]

# Problem sizes
n_values = np.arange(5000, 50001, 5000)

# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(8,6))

for r in r_values:

    data = np.load(f"relative_errors_r_{r}.npz")

    rel_err = data["rel_err"]
    print(rel_err)

    plt.plot(
        n_values,
        rel_err,
        marker='o',
        linewidth=2,
        label=f"r = {r}"
    )

plt.xlabel("Number of particles (n)")
plt.ylabel("Relative Error")
plt.title("Relative Error of FMM Roots")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()


# ------------------ PLOTTING AS SUBPLOTS ------------------------
# =====================================================
# Ranks to plot
# =====================================================

r_values = [30, 40, 50, 60]

# Problem sizes
n_values = np.arange(5000, 50001, 5000)

# =====================================================
# Create 2x2 subplots
# =====================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

axes = axes.flatten()

for ax, r in zip(axes, r_values):

    data = np.load(f"relative_errors_r_{r}.npz")
    rel_err = data["rel_err"]

    ax.plot(
        n_values,
        rel_err,
        marker='o',
        linewidth=2
    )

    # Annotate each point
    for x, y in zip(n_values, rel_err):
        ax.text(
            x,
            y,
            f"{y:.2e}",
            fontsize=8,
            ha='center',
            va='bottom'
        )

    ax.set_title(f"r = {r}")
    ax.set_xlabel("n")
    ax.set_ylabel("Relative Error")
    ax.grid(True)

fig.suptitle("Relative Error of FMM Roots", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()

#-------------------------- INTERLACING CHECK-------------------
#"""

from scipy.io import loadmat
import os
import sys


project_root = os.path.abspath(".")
if project_root not in sys.path:
    sys.path.append(project_root)

# Load the original data
data = loadmat("data2.mat")
data2 = data["data2"]

# Load the FMM results
results = np.load("fmm_random_results_r_20.npz", allow_pickle=True)
fmm_y = results["fmm_y"]
exact_y       = results["exact_y"]

# Example: n = 5000 (jj = 0)
jj = 0

x = np.squeeze(data2[jj, 0])   # source points
y = fmm_y[jj]                  # computed roots

check_interlacing_sigma_pos(x, y, tol=1e-12)












