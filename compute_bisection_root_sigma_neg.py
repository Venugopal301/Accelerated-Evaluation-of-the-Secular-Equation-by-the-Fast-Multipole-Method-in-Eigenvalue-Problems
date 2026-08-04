#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:09:46 2026

@author: venugopal
"""

import numpy as np

def compute_bisection_root_sigma_neg(x, q, sigma, tol):

    n = len(x)

    y0 = np.empty(n)

    eps = 1e-12

    # -------------------------------------------------------
    # First root: left of x[0]
    # -------------------------------------------------------
    gap = x[1] - x[0]
    
    b = x[0] - eps
    a = x[0] - 5.0 * gap
    
    while secular(a, x, q, sigma) * secular(b, x, q, sigma) > 0:
        a -= 5.0 * gap

    y0[0] = bisection_root(
        x, q, sigma,
        a, b,
        tol=tol
    )

    # -------------------------------------------------------
    # Remaining roots: x[i-1] < y[i] < x[i]
    # -------------------------------------------------------

    for i in range(1, n):

        a = x[i-1] + eps
        b = x[i] - eps

        y0[i] = bisection_root(
            x, q, sigma,
            a, b,
            tol=tol
        )

    return y0


def bisection_root(x, q, sigma, a, b,
                   tol, max_iter=100):

    fa = secular(a, x, q, sigma)
    fb = secular(b, x, q, sigma)

    if fa * fb > 0:
        print(f"Warning: No sign change in [{a:.6e}, {b:.6e}]. Using midpoint.")
        return 0.5 * (a + b)

    for _ in range(max_iter):

        c = 0.5 * (a + b)
        fc = secular(c, x, q, sigma)

        # Residual convergence
        if abs(fc) < 1e-13:
            return c

        # Interval convergence
        if abs(b - a) < tol:
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return 0.5 * (a + b)


def secular(y, x, q, sigma):
    return 1.0 + sigma * np.sum(q**2 / (x - y))