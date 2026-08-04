#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:56:21 2026

@author: venugopal

Test data from:  Tang, Tao and Yang, Jiang, Finding the Maximal Eigenpair for a Large, Dense, Symmetric
Matrix based on Mufa Chen’s Algorithm, 2020, https://doi.org/10.4208/cmr.2020-0005.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
#from scipy.sparse import issparse
#from scipy.sparse import diags
#import matplotlib.pyplot as plt
from lanczos_full_reorthogonalization import lanczos_full_reorthogonalization
from scipy.linalg import block_diag
from scipy.sparse.linalg import eigsh
from scipy.linalg import block_diag



def signed_gauss_laguerre_matrix(N, alpha=-0.25):
    """
    Construct the signed Gauss-Laguerre tridiagonal matrix G_{N+1}^s.

    Parameters
    ----------
    N : int
        Matrix size will be (N+1)x(N+1).
    alpha : float
        Laguerre parameter (default = -0.25)

    Returns
    -------
    G : ndarray
        Symmetric signed Gauss-Laguerre matrix.
    """

    n = N + 1

    # Diagonal entries
    diag = np.array([2*i + 1 + alpha for i in range(n)], dtype=float)

    # Off-diagonal entries
    offdiag = np.zeros(n-1)

    for i in range(n-1):
        beta = np.sqrt((i+1)*(i+1+alpha))
        gamma = (-1)**i
        offdiag[i] = gamma * beta

    # Assemble tridiagonal matrix
    G = np.diag(diag)
    G += np.diag(offdiag, k=1)
    G += np.diag(offdiag, k=-1)

    return G


# ---------------- Example ----------------

N = 99
G = signed_gauss_laguerre_matrix(N)

print("Matrix size :", G.shape)

"""
k = 2000  # size of the tridiagonal matrix
alpha, beta, Q, T = lanczos_full_reorthogonalization(G, k)
print(T.shape)

I = Q.T @ Q

print(np.linalg.norm(I - np.eye(I.shape[0])))

T_dense = T.toarray()

#err = np.linalg.norm(G @ Q - Q @ T_dense)

#print(err)
"""

# Main diagonal
alpha = np.diag(G)

# Superdiagonal
beta = np.diag(G, k=1)

n = len(alpha)
m = n // 2

#------------------ deriving the secular equation parameters

alpha1 = alpha[:m].copy()
beta1  = beta[:m-1].copy()

bm = beta[m-1]

alpha1[-1] -= bm

alpha2 = alpha[m:].copy()
beta2  = beta[m:].copy()

alpha2[0] -= bm

Lambda1, Q1 = eigh_tridiagonal(alpha1, beta1)

Lambda2, Q2 = eigh_tridiagonal(alpha2, beta2)

v = np.zeros(len(alpha))

v[m-1] = 1.0
v[m]   = 1.0

Q = block_diag(Q1,Q2)

u = Q.T @ v

Lambda = np.concatenate((Lambda1,Lambda2))


""" 
#verify the T constructed
M = np.diag(Lambda) + bm*np.outer(u,u)
T_reconstructed = Q @ M @ Q.T
err = np.linalg.norm(G - T_reconstructed)

print(err)

err_abs = np.linalg.norm(T_dense - T_reconstructed, ord='fro')

err_rel = err_abs / np.linalg.norm(T_dense, ord='fro')

print("Absolute error :", err_abs)
print("Relative error :", err_rel)
"""

perm = np.argsort(Lambda)
Lambda_sorted = Lambda[perm]
u_sorted = u[perm]

d = Lambda_sorted.copy()
q = u_sorted
sigma = bm

#eig_T = eigh_tridiagonal(alpha, beta, eigvals_only=True)

#----------------- save the variables
np.savez(
    "pde_2_data_100.npz",
    d=d,
    q=q,
    sigma=sigma
  
)

print("Data saved successfully.")

