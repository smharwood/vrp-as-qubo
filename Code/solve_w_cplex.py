"""
14 June 2022
SM Harwood

Solve various versions of the routing problems with CPLEX
"""
import os
import numpy as np
import TestTools as TT
try:
    import cplex
    have_cplex = True
except ImportError:
    have_cplex = False
    print("No CPLEX; This prolly won't work...")

def main(testset_path):
    """
    Go thru all files in path, load the *.rudy's and *.lp's, solve
    """
    fnames = os.listdir(testset_path)
    ising_matrices = []
    for f in fnames:
        if f.split('.')[1] == ".rudy":
            ising_matrices.append(TT.loadIsingMatrix(f))
            # Convert to QUBO?
            # make CPLEX object?
    return