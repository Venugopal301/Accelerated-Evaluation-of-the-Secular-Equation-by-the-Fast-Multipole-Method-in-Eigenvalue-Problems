# Accelerated Evaluation of the Secular Equation by the Fast Multipole Method in Eigenvalue Problems

This repository contains a Python implementation of a one-dimensional Fast Multipole Method (FMM) for the efficient evaluation of the secular equation arising from symmetricxranyodifications of diagonal matrices. The implementation is designed to accelerate the repeated evaluation of

[
f(\lambda)=1+\sigma\sum_{i=1}^{n}\frac{q_i}{x_i-y},
]

which appears in divide-and-conquer eigensolvers and related numerical linear algebra algorithms.

The repository includes the complete hierarchical tree construction, multipole and local expansion operators, upward and downward passes, near-field corrections, Newton iteration for root computation, and performance analysis utilities.

---

## Features

- Adaptive one-dimensional hierarchical tree construction.
- Fast Multipole Method (FMM) implementation for the kernel
  [
  K(x,y)=\frac{1}{x-y}.
  ]
- Newton iteration for solving the secular equation.
- Direct evaluation for accuracy verification.
- Relative error analysis.
- Timing comparisons between FMM and direct evaluation.
- FLOP count estimation.
- Scripts for generating numerical experiments and figures.

---

## Requirements

The implementation requires:

- Python 3.10.12
- NumPy
- SciPy
- Matplotlib

Install the required packages using

```bash
pip install numpy scipy matplotlib
```

---

## Running the Code

Run the main Python script

```bash
python fmm_random_data.py
python FMM_pde_2.py
```

The program constructs the adaptive FMM tree, performs Newton iterations using both the FMM and direct evaluation, and reports timing, accuracy, and relative error statistics.

---

## Output

Typical outputs include

- Newton iteration timings
- Function residuals
- Relative errors
- FLOP counts
- Saved `.npz` files for post-processing
- Publication-quality figures

---

## Applications

This implementation is intended for research in

- Numerical Linear Algebra
- Eigenvalue Problems
- Rank-One Matrix Modifications
- Fast Multipole Methods
- Scientific Computing

---

## Citation

If you use this repository in your research, please cite the associated publication once available.

---

## License

This repository is intended for academic and research purposes.
