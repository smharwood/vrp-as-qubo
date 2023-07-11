# MIRP as QUBO:

Casting Vehicle Routing Problems (VRP), and specifically Maritime Inventory Routing Problems (MIRP), as Quadratic Unconstrained Binary Optimization problems (QUBO)

## Installation
After cloning the repo, install as, e.g.
```
pip install .
```
or without cloning:
```
pip install vrpqubo @ git+https://github.com/smharwood/mirpasqubo
```
However, the package is pretty simple and after cloning everything should work if your working directory is `src/`.

## Dependencies
This package has minimal dependencies (`scipy`).
However, some features rely on a full installation of the CPLEX solver.

## Generating a Test Set
One goal of this package is to generate test problems in standard forms (QUBO or Ising).
After installation, the script `generate_test_set(.exe)` will be on the path.
Running
```
generate_test_set -p TestSet -t 20 30 40 50
```
generates a test set of various sized problems in the folder `TestSet`.  
This will include "feasibility" versions (minimum objective value equals zero) of each optimization problem as well.

See `src/vrpqubo/examples/mirp_g1.py` as a template to develop your own examples.

## Source structure
### doc
Detailed (but incomplete) mathematics of formulations and ideas in code

### src
Source code for package and tests