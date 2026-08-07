#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:59:08 2026

@author: venugopal
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Load saved results
# ==========================================

data = np.load("pde_2_results_2000.npz", allow_pickle=True)

q = data["q"]
x = data["x"]
fmm_y = data["fmm_y"]
exact_y = data["exact_y"]

# ==========================================
# Indices satisfying q > 1e-16
# ==========================================

tol = 1e-16
idx = np.where(q > tol)[0]

print(f"Number of q values > {tol}: {len(idx)}")

# ==========================================
# Relative error for selected roots
# ==========================================

eps = np.finfo(float).eps

rel_err = np.abs(fmm_y[idx] - exact_y[idx]) / np.maximum(np.abs(exact_y[idx]), eps)

print("\nRelative errors:")
for i, err in zip(idx, rel_err):
    print(f"Index {i:5d}: q = {q[i]:.3e}, Relative Error = {err:.3e}")

# Overall relative error
overall_rel_err = (
    np.linalg.norm(fmm_y[idx] - exact_y[idx])
    / np.linalg.norm(exact_y[idx])
)

print("\nOverall relative error =", overall_rel_err)

#---------------------- PLOTTING THE RELATIVE ERRORS ------------------

plt.figure(figsize=(10,5))

plt.semilogy(
    idx,
    rel_err,
    marker='o',
    linewidth=1.5,
    markersize=4
)

plt.xlabel("Root Index")
plt.ylabel("Relative Error")
plt.title(r"Relative Error for Roots with $q > 10^{-16}$")
plt.grid(True, which='both')

plt.tight_layout()
plt.show()
#-------------------------------------------
# plotting q vs accuracy

plt.figure(figsize=(10,5))

plt.loglog(
    q[idx],
    rel_err,
    marker='o',
    linestyle='None',
    markersize=4
)

plt.xlabel(r"$q_i$")
plt.ylabel("Relative Error")
plt.title(r"Relative Error vs. $q_i$ for $q_i > 10^{-16}$")
plt.grid(True, which='both')

plt.tight_layout()
plt.show()