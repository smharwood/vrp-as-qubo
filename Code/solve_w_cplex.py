"""
14 June 2022
SM Harwood

Solve various versions of the routing problems with CPLEX
Note that we are ignoring the original formulation with hard constraints - 
given an Ising problem/QUBO, how well do classical methods handle it?
"""
import os, argparse
import cplex
import numpy as np
import scipy.sparse as sp
import TestTools as TT
import QUBOTools as QT

def main():
    parser = argparse.ArgumentParser(description=
        "Solve Ising problems and binary ILP with CPLEX",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p','--prefix', type=str, default='.',
                        help="Folder to look for problem files")
    parser.add_argument('-i','--ising', action="store_true",
                        help="Look for and solve all Ising problems")
    parser.add_argument('-l','--lp', action="store_true",
                        help="Look for and solve all ILP problems")
    parser.add_argument('-v','--verbose', action="store_true",
                        help="Be verbose")
    args = parser.parse_args()

    no_action = True
    if args.ising:
        solve_all_isings(args.prefix, args.verbose)
        no_action = False
    if args.lp:
        solve_all_lps(args.prefix, args.verbose)
        no_action = False
    if no_action:
        parser.print_help()
    return

def solve_all_isings(testset_path, verbose=True):
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
            J, h = QT.get_Ising_J_h(mat)
            Q, c = QT.Ising_to_QUBO(J, h, c)
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
            xstar = cplex_prob.solution.get_values()
            sol_fn = os.path.splitext(f)[0] + ".sol"
            sol_path = os.path.join(testset_path, sol_fn)
            with open(sol_path, 'w') as s:
                spins = QT.x_to_s(np.asarray(xstar))
                for spin in spins:
                    s.write("{}\n".format(int(spin)))
            if verbose:
                print("Instance {}, optimal objective value {} found in {} seconds".format(f, objective, soltime))
    return optimal_objectives, solution_times

def solve_all_lps(testset_path, verbose=True):
    """ Solve all .lp's in a directory with CPLEX """
    fnames = os.listdir(testset_path)
    optimal_objectives = dict()
    solution_times = dict()
    for f in fnames:
        if f.split('.')[-1] == "lp":
            if verbose: print("Processing {}...".format(f))
            # Found a `.lp`
            # Load and solve
            cplex_prob = cplex.Cplex(os.path.join(testset_path, f))
            if not verbose:
                cplex_prob.set_log_stream(None)
                cplex_prob.set_results_stream(None)
            start = cplex_prob.get_time()
            cplex_prob.solve()
            soltime = cplex_prob.get_time() - start
            stat = cplex_prob.solution.get_status_string()
            if stat.lower() == "integer optimal solution":
                objective = cplex_prob.solution.get_objective_value()
            else:
                objective = np.inf
                if verbose:
                    print("Instance {}, status {}".format(f, stat))
            optimal_objectives[f] = objective
            solution_times[f] = soltime
            # TODO: time to first solution??
            # NOTE: solution values are probably meaningless
            # I think reading from the .lp file messes with the variable order
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

if __name__ == "__main__":
    main()