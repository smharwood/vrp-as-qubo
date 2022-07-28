"""
10 September 2021
SM Harwood

Tool to test feasibility of spins in original (hard-constrained) problem
"""
import os, argparse
import numpy as np
import scipy.sparse as sp
from QUBOTools import s_to_x
from TestTools import loadSpins

def test_feasibility(x, A_eq, b_eq, Q_eq, r_eq):
    """
    Given 0-1 variables and constraints definition, return measures of the
    violated constraints
    A_eq * x = b_eq
    xᵀ * Q_eq * x = r_eq

    Parameters:
    x (array): 1-d array of 0-1 (binary) variables to test
    A_eq, b_eq: Array and vector defining equality constraints
    Q_eq, r_eq: Array and scalar defining inequality constraints

    Returns:
    vio_l (array): 1-d boolean array of violated linear constraints
        vio_l = (A_eq.dot(x) != b_eq)
    vio_q (float): A scalar measure of the violation of the quadratic constraint
        vio_q = x.transpose().dot(Q_eq).dot(x) - r_eq
    Q_nnz (int): The number of nonzero elements in Q_eq
        (the quadratic constraint might actually represent multiple constraints,
        in which case the number of nonzero elements is the "original" number
        of constraints)
    """
    vio_l = (A_eq.dot(x) != b_eq)
    vio_q = np.dot(x, Q_eq.dot(x)) - r_eq
    return vio_l, vio_q, Q_eq.nnz

# TODO: test
def convenience(fname, sname):
    spins = loadSpins(sname)
    x = s_to_x(spins)
    f = np.load(fname, allow_pickle=True)
    A_eq = f["A_eq"]
    b_eq = f["b_eq"]
    Q_eq = f["Q_eq"]
    r_eq = f["r_eq"]
    # see generate_test_set.py - we used np.savez;
    # If there are linear equality constraints,
    # we might be loading a length-1 array of a sparse matrix object;
    # pull out the right stuff
    if len(b_eq) > 0:
        if sp.issparse(A_eq.item(0)):
            A_eq = A_eq.item(0)
    # Similarly with the quadratic constraint;
    # we assume that the arrays have dimensions greater than zero
    if sp.issparse(Q_eq.item(0)):
        Q_eq = Q_eq.item(0)
    return test_feasibility(x, A_eq, b_eq, Q_eq, r_eq)

def do_all(prefix, verbose=True):
    """
    Find all foo.npz and check the feasibility of foo.sol,
    if it exists
    """
    fnames = os.listdir(prefix)
    results = dict()
    for f in fnames:
        if f.split('.')[-1] == "npz":
            if verbose: print("Processing {}...".format(f))
            sol_fn = os.path.splitext(f)[0] + ".sol"
            sol_path = os.path.join(prefix, sol_fn)
            f_path = os.path.join(prefix, f)
            if os.path.isfile(sol_path):
                results[f] = convenience(f_path, sol_path)
            elif verbose:
                print("Could not find corresponding solution file {}".format(sol_path))
    return results

def main():
    parser = argparse.ArgumentParser(description=
        "Test feasibility of spins\n\n"+
        "Test set should include *.npz files; "+
        "these include the original problem constraints as linear and "+
        "quadratic constraints.\n"+
        "This script will return a measure of the "+
        "linear and quadratic (if any) constraints that are violated",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d','--data', type=str,
                        help="Filename of feasibility data ('*.npz')")
    parser.add_argument('-s','--spins', type=str,
                        help="Filename of spins\n(whitespace- and/or linebreak-separated)")
    args = parser.parse_args()
    if args.data is None or args.spins is None:
        parser.print_help()
        return
    vio_l, vio_q, nnz = convenience(args.data, args.spins)
    print("Number of UNsatisfied constraints:")
    print("Linear:    {} out of {}".format(sum(vio_l), len(vio_l)))
    print("Quadratic: {} out of {}".format(int(vio_q), nnz))

if __name__ == "__main__":
    main()