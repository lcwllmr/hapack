# `hapack`: Solving (skew-)Hamiltonian eigenvalue problems

[![ci](https://github.com/lcwllmr/hapack/actions/workflows/ci.yml/badge.svg)](https://github.com/lcwllmr/hapack/actions/workflows/ci.yml)

This Python package contains Python re-implementations of the routines presented in the article `[BK06]` above using only `numpy` and `scipy`.
The original Fortran 77 source by the authors is available as algorithm 854 at [CALGO](https://calgo.acm.org/).
Some helper routines are taken from `[VG96]`.

References:

- `[BK06]`: Benner, P. and Kressner, D., 2006.
    *Algorithm 854: Fortran 77 subroutines for computing the eigenvalues of Hamiltonian matrices II*.
    ACM Transactions on Mathematical Software (TOMS), 32(2), pp.352-373.
    [doi:10.1145/1141885.1141895](https://doi.org/10.1145/1141885.1141895).
- `[VG96]`: van Loan, C. F. and Golub, G. H., 1996.
    *Matrix Computations* (3rd ed.).
    JHU press.
    ISBN 0-8018-5414-8.

## Local development

Make sure that [uv](https://docs.astral.sh/uv/) is installed on the system and run `uv sync` to set up the virtual environment.
To install the pre-commit hooks (format, lint, test) run `uvx pre-commit install`.
Tests can be run with `uv run pytest`.

## Changelog

**v0.1.0** (WIP):

- un-optimized but clean and correct implementations of all algorithms in `[BK06]`
- 100% test coverage and basic ci with pypi publishing
- simple documentation for each algorithm: input, output and where to find it in the paper.
