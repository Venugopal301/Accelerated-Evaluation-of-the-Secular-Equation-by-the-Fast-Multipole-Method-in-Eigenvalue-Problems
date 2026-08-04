# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:32:01 2026

@author: se22pmat003
"""

import numpy as np
from scipy.io import loadmat
import time
import os
import sys

from data_structure import data_structure
from particle_to_local import particle_to_local
from flops import flops

project_root = os.path.abspath(".")
if project_root not in sys.path:
    sys.path.append(project_root)

# Load data2.mat
data = loadmat("data2.mat")

# Extract data2
data2 = data["data2"]

# Preallocate containers
setup_time = np.zeros((10, 1))
elapsed_time = [[None for _ in range(2)] for _ in range(10)]
fmm_fun_val = [None for _ in range(10)]
fmm_y = [None for _ in range(10)]
flop_counts = [None for _ in range(10)]

# Parameters
scaling = 0
max_iter = 10  # Maximum Newton iterations

r = 60

#"""
for jj in range(10):

    print(jj)

    x = data2[jj, 0]  # sources
    y = data2[jj, 1]  # targets
    q = data2[jj, 2]  # charges

    x = np.squeeze(x)
    y = np.squeeze(y)
    q = np.squeeze(q)

    q = q**2  # elementwise square

    start_time = time.perf_counter()

    u1, u2, post_ord, ch, Px, Py, I, nflops, neighbor = data_structure(r, x, y, q, scaling)


    setup_time[jj] = time.perf_counter() - start_time

    fmm_timer = np.zeros(max_iter)
      
    for iter in range(max_iter):

        ftt = time.perf_counter()

        z1 = np.zeros(len(q))
        z2 = np.zeros(len(q))

        # ==========================================
        # Near-field interaction & final evaluation
        # ==========================================

        for i in post_ord:
            idx = i - 1
            if len(ch[idx]) == 0:
                py = Py[idx]
                if len(py) > 0:
                    yj = y[py]
                    if u1[idx] is not None:
                        Ui = particle_to_local(yj, r, I[idx, 0], 2 * I[idx, 1], scaling)
                        z1[py] = Ui @ u1[idx]
                        z2[py] = Ui @ u2[idx]
                        nflops += flops("prod", Ui, "n", u1[idx], "n")
                        near_nodes = [i] + neighbor[idx]

                    for j in np.unique(near_nodes):
                        jdx = j - 1
                        px = Px[jdx]
                        if len(px) > 0:
                            xi = x[px]
                            D1 = 1.0 / (xi[np.newaxis, :] - yj[:, np.newaxis])
                            D2 = 1.0 / (xi[np.newaxis, :] - yj[:, np.newaxis]) ** 2
                            D1[np.isinf(D1)] = 0.0
                            D2[np.isinf(D2)] = 0.0
                            z1[py] += D1 @ q[px]
                            z2[py] += D2 @ q[px]
                            nflops += D1.shape[0] * (2 * D1.shape[1] - 1) + len(py)

        w1 = 1.0 + z1
        ynew = y - (w1 / z2)
        
        y = ynew
         
        
        fmm_timer[iter] = time.perf_counter() - ftt

    fmm_fun_val[jj] = w1.copy()  # store function values
    fmm_y[jj] = ynew.copy()  # store updated targets

    elapsed_time[jj][0] = fmm_timer  # first column

    flop_counts[jj] = nflops

#"""
#
# ==========================================
# Direct evaluation of secular equation
# ==========================================
#
"""
exact_fun_val = [None] * 10
exact_y = [None] * 10


# ==========================================
# Data loop
# ==========================================

for jj in range(10):

    print(jj)

    x = data2[jj][0]  # sources
    y = data2[jj][1]  # targets
    q = data2[jj][2]

    x = np.squeeze(x)
    y = np.squeeze(y)
    q = np.squeeze(q)

    q = q**2

    n = len(y)

    exact_timer = np.zeros(max_iter)

    # ======================================
    # Newton iterations
    # ======================================
    
    for iter in range(max_iter):

        dtt = time.perf_counter()
        
        exact_fun = np.zeros(n, dtype=np.float64)
        exact_der = np.zeros(n, dtype=np.float64)

        for k in range(n):

            dn = x - y[k]  # (x-y)
            der_dn = dn**2  # (x-y)^2

            exact_fun[k] = np.sum(q / dn)

            exact_der[k] = np.sum(q / der_dn)

        v1 = 1.0 + exact_fun

        ynew = y - (v1 / exact_der)
        
        y = ynew
        
        exact_timer[iter] = time.perf_counter() - dtt

    # ======================================
    # Store results
    # ======================================

    exact_fun_val[jj] = v1.copy()

    exact_y[jj] = ynew.copy()

    elapsed_time[jj][1] = exact_timer
"""


# ==========================================
# Save all results
# ==========================================

np.savez(
    "fmm_random_results_r_60.npz",
    setup_time=setup_time,
    #elapsed_time=np.array(elapsed_time, dtype=object),
    fmm_fun_val=np.array(fmm_fun_val, dtype=object),
    fmm_y=np.array(fmm_y, dtype=object),
    flop_counts=np.array(flop_counts, dtype=object),
    #exact_fun_val=np.array(exact_fun_val, dtype=object),
    #exact_y=np.array(exact_y, dtype=object),
    allow_pickle=True
)

print("Results saved successfully")
