"""
10 September 2021
SM Harwood

Tool to test feasibility of spins in original (hard-constrained) problem
"""
import argparse
import numpy as np
from QUBOTools import s_to_x
from TestTools import loadSpins

def test_feasibility(x, A_eq, b_eq, A_ineq, b_ineq):
    """
    Given 0-1 variables and constraints definition, return indices of 
    violated constraints
    A_eq * x = b_eq
    A_ineq * x \le b_ineq

    Parameters:
    x (array): 1-d array of 0-1 (binary) variables to test
    A_eq, b_eq: Array and vector defining equality constraints
    A_ineq, b_ineq: Array and vector defining inequality constraints

    Returns:
    vio_eq (array): 1-d boolean array of violated equality constraints
        vio_eq = (A_eq.dot(x) == b_eq)
    vio_ineq (array): 1-d boolean array of violated inequality constraints
        vio_ineq = (A_ineq.dot(x) <= b_ineq)
    """
    vio_eq = (A_eq.dot(x) == b_eq)
    vio_ineq = (A_ineq.dot(x) <= b_ineq)
    return vio_eq, vio_ineq

def convenience(fname, sname):
    f = np.load(fname)
    spins = loadSpins(sname)
    x = s_to_x(spins)
    return test_feasibility(x, f['A_eq'], f["b_eq"], f["A_ineq"], f["b_ineq"])

def main():
    parser = argparse.ArgumentParser(description=
        "Test feasibility of spins\n\n"+
        "Test set should include *.npz files; "+
        "these include the original problem constraints as linear constraints.\n"+
        "This script will return a Boolean array of the "+
        "equality and inequality (if any) constraints that are violated",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d','--data', type=str,
                        help="Filename of feasibility data ('*.npz')")
    parser.add_argument('-s','--spins', type=str,
                        help="Filename of spins\n(Either whitespace-separated on one line, or one entry per line")
    args = parser.parse_args()
    if args.data is None or args.spins is None:
        parser.print_help()
        return
    convenience(args.data, args.spins)

if __name__ == "__main__":
    main()