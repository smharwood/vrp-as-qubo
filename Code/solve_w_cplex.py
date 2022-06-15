"""
14 June 2022
SM Harwood

Solve various versions of the routing problems with CPLEX
Note that we are ignoring the original formulation with hard constraints - 
given an Ising problem/QUBO, how well do classical methods handle it?
"""
import os
import cplex
import scipy.sparse as sp
import TestTools as TT

def main(testset_path, verbose=True):
    """
    Find all the Ising problems in the path (*.rudy's),
    solve with CPLEX
    """
    fnames = os.listdir(testset_path)
    optimal_objectives = dict()
    solution_times = dict()
    for f in fnames:
        if f.split('.')[-1] == "rudy":
            if verbose: print("Processing {}...".format(f))
            # Found an Ising problem;
            # Load the matrix,
            # extract the correct representation,
            # convert to QUBO (so we get a problem with 0-1 variables)
            # construct the relevant CPLEX object
            mat, c = TT.loadIsingMatrix(os.path.join(testset_path, f))
            J, h = TT.get_ising_j_h(mat)
            Q, c = TT.Ising_to_QUBO_sparse(J, h, c)
            cplex_prob = build_cplex_from_qubo(Q)
            start = cplex_prob.get_time()
            cplex_prob.solve()
            soltime = cplex_prob.get_time() - start
            raw_objective = cplex_prob.solution.get_objective_value()
            objective = raw_objective + c
            optimal_objectives[f] = objective
            solution_times[f] = soltime
            # TODO: time to first solution??
            # Export qubo as .lp so we can solve with other solvers?
            if verbose:
                print("Instance {}, optimal objective value {} found in {} seconds".format(f, objective, soltime))
    return optimal_objectives, solution_times

def build_cplex_from_qubo(Q):
    """ Get a CPLEX object for the QUBO """
    assert sp.issparse(Q), "Expecting a sparse matrix input"

    # num vars
    n = Q.shape[0]

    # copy out diagonal, then set to zero
    Qcop = sp.coo_matrix(Q)
    c = Q.diagonal().tolist()
    Qcop.setdiag(0)

    # Define object
    cplex_prob = cplex.Cplex()
    cplex_prob.objective.set_sense(cplex_prob.objective.sense.minimize)
    
    # Variables: all binary
    var_types = [cplex_prob.variables.type.binary] * n
    cplex_prob.variables.add(obj=c, types=var_types)

    # Quadratic objective
    rows = Qcop.row.tolist()
    cols = Qcop.col.tolist()
    vals = Qcop.data.tolist()
    cplex_prob.objective.set_quadratic_coefficients(zip(rows,cols,vals))
    return cplex_prob