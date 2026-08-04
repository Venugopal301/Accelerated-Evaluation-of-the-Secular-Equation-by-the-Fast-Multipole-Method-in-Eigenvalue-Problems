#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 10:07:40 2026

@author: venugopal
Test data from:  Tang, Tao and Yang, Jiang, Finding the Maximal Eigenpair for a Large, Dense, Symmetric
Matrix based on Mufa Chen’s Algorithm, 2020, https://doi.org/10.4208/cmr.2020-0005.
"""

import numpy as np
import time
import os
import sys


import matplotlib.pyplot as plt

from data_structure import data_structure
from particle_to_local import particle_to_local
from flops import flops
from check_interlacing_sigma_neg import check_interlacing_sigma_neg
from compute_bisection_root_sigma_neg import compute_bisection_root_sigma_neg
#from scipy.linalg import eigvalsh
#from secular_bisection_negative import secular_bisection_negative

project_root = os.path.abspath(".")
if project_root not in sys.path:
    sys.path.append(project_root)

from pathlib import Path

data_file = Path("test_data_pde") / "pde_2_data_1999.npz"

data = np.load(data_file)


x = data["d"]
q = data["q"]
sigma = float(data["sigma"])

n = len(x)

def secular(y, x, q, sigma):
    return 1.0 + sigma * np.sum(q**2 / (x - y))

def secular_derivative(y, x, q, sigma):
    return sigma*np.sum(q**2/(x-y)**2)

# Initial guesses
y_init = compute_bisection_root_sigma_neg(x, q, sigma, tol = 1e-3)
y = y_init
"""
D = np.diag(x)
A = D + sigma * np.outer(q, q)

# Compute eigenvalues
eigvals = np.linalg.eigvalsh(A)

print("Number of eigenvalues:", len(eigvals))
print("First 10 eigenvalues:")
print(eigvals[:10])

y = eigvals
ok_initial, initial_interlace_fail, _ = check_interlacing_sigma_neg(x, y)
"""
# Function values at the roots computed by bisection
fun_val_init = np.array([secular(y, x, q, sigma) for y in y])

eig_G = compute_bisection_root_sigma_neg(x, q, sigma, tol = 1e-14)
ok_eig, eig_interlace_fail, _ = check_interlacing_sigma_neg(x, eig_G)

fun_val_eig = np.array([secular(y, x, q, sigma) for y in eig_G])

bad = np.where(fun_val_eig > 1e-10)[0]


#----------------------------------------------------------------------
# Preallocate containers

setup_time = np.zeros((10, 1))
elapsed_time = [[None for _ in range(2)] for _ in range(10)]
fmm_fun_val = [None for _ in range(10)]
fmm_y = [None for _ in range(10)]
flop_counts = [None for _ in range(10)]

# Parameters
scaling = 0
max_iter = 10  # Maximum Newton iterations
tol = 1e-6

#---------------------FMM computation --------
r = 20


x = np.squeeze(x)
y = np.squeeze(y)
q = np.squeeze(q)

u = q
q = q**2  # elementwise square

start_time = time.perf_counter()

u1, u2, post_ord, ch, Px, Py, I, nflops, neighbor = data_structure(r, x, y, q, scaling)


setup_time = time.perf_counter() - start_time

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

    w1 = 1.0 + sigma*z1
    w2 = sigma*z2
    ynew = y - (w1 / w2)
    y = ynew
        
    fmm_timer[iter] = time.perf_counter() - ftt

fmm_fun_val = w1.copy()  # store function values
fmm_y = ynew.copy()  # store updated targets

elapsed_time[0] = fmm_timer  # first column

flop_counts = nflops

#-------------------------end of FMM computation ----------------------------


# ==========================================
# Direct evaluation of secular equation
# ==========================================

# Initial guesses
y = y_init

x = np.squeeze(x)
y = np.squeeze(y)

exact_timer = np.zeros(max_iter)

# ======================================
# Newton iterations
# ======================================
ne = len(x)

for iter in range(max_iter):

    dtt = time.perf_counter()
    
    exact_fun = np.zeros(ne, dtype=np.float64)
    exact_der = np.zeros(ne, dtype=np.float64)

    for k in range(ne):

        dn = x - y[k]  # (x-y)
        der_dn = dn**2  # (x-y)^2

        exact_fun[k] = np.sum(q / dn)

        exact_der[k] = np.sum(q / der_dn)

    v1 = 1.0 + sigma*exact_fun
    v2 = sigma*exact_der

    ynew = y - (v1 / v2)
    y = ynew
    
    exact_timer[iter] = time.perf_counter() - dtt

# ======================================
# Store results
# ======================================

exact_fun_val = v1.copy()
exact_y = ynew.copy()
elapsed_time[1] = exact_timer

#------------------------------------ end of Exact computation ------------------
#---------------------------- save results

np.savez(
    "pde_2_results_2000.npz",
    elapsed_time_1=np.array(elapsed_time[0]),
    elapsed_time_2=np.array(elapsed_time[1]),
    fmm_fun_val=fmm_fun_val,
    fmm_y=fmm_y,
    exact_fun_val=exact_fun_val,
    exact_y=exact_y,
    q=q,
    u=u,
    x = x,
    y=y,
    sigma=sigma,
    r= r
)

print("Results saved successfully.")








