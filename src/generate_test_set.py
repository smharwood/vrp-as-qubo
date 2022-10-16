# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:37:45 2019

@author: smharwo

Test instances for Stanford/Cornell:
vary TimeHorizon and number of routes added to get different size problems
"""
import argparse, os
import numpy as np
from itertools import product
from QUBOTools import QUBOContainer, x_to_s
import arc_based.ExMIRPg1 as abex
import path_based.ExMIRPg1 as pbex
import sequence_based.ExMIRPg1 as sbex
try:
    import cplex
    from solve_w_cplex import CPLEX_FEASIBLE
    have_cplex = True
except ImportError:
    have_cplex = False


def gen(prefix, horizons):
    """ Generate test set """
    formulations = [abex, pbex, sbex]
    for (TH, ex) in product(horizons, formulations):
        mod = ex.__name__.split('.')[0]
        name = ''.join([w[0] for w in mod.split('_')])

        # define problem and export it
        prob = ex.DefineProblem(TH, make_feasible=True)
        
        # Version with objective:
        Q, c = prob.getQUBO(None, feasibility=False)
        QC = QUBOContainer(Q, c)
        print("Time horizon {}, formulation {}, number of variables: {}".
            format(TH, mod, prob.getNumVariables()))
        bname = "test_{}_{}_".format(name, prob.getNumVariables())
        bname = os.path.join(prefix, bname)
        # ising_o_name = 
        QC.export(bname + "o.rudy", as_ising=True)

        # Feasibility version
        Q, c = prob.getQUBO(None, feasibility=True)
        QC = QUBOContainer(Q, c)
        # ising_f_name = 
        QC.export(bname + "f.rudy", as_ising=True)

        # export constraint set
        # Note that some of these matrices might be scipy.sparse,
        # in which case savez is not the most natural way to save them...
        # but we can hack our way around it
        A_eq, b_eq, Q_eq, r_eq = prob.getConstraintData()
        np.savez(bname, A_eq=A_eq, b_eq=b_eq, Q_eq=Q_eq, r_eq=r_eq)

        if have_cplex:
            # We should solve the problem now - 
            # variable order might get messed up reading from the .lp
            # Also, focus on finding feasible solution to test other features
            prob.export_mip(bname + "o.lp")
            cplex_prob = prob.getCplexProb()
            cplex_prob.parameters.mip.limits.solutions.set(1)
            cplex_prob.solve()
            stat = cplex_prob.solution.get_status_string()
            if stat.lower() not in CPLEX_FEASIBLE:
                print("No solution written; status: {}".format(stat))
                continue
            xstar = cplex_prob.solution.get_values()
            sol_path = bname + ".sol"
            with open(sol_path, 'w') as s:
                spins = x_to_s(np.asarray(xstar))
                for spin in spins:
                    s.write("{}\n".format(int(spin)))
    return

def main():
    parser = argparse.ArgumentParser(description=
        "Build a set of test problems\n\n"+
        "For example, running\n"+
        "python generate_test_set.py -p TestSet -t 20 30 40 50\n"+
        "builds a test set in the folder \"TestSet\" with 24 total instances\n"+
        "(\"feasibility\" and \"optimality\" versions for each of three different formulations\n"+
        "for each of the four time horizons given)\n\n"+
        "Each file has the name \"test_<formulation>_<size>_<class>.rudy\"\n"+
        "where <formulation> indicates which formulation is used,\n"+
        "      <size> indicates the number of variables,\n"+
        "      <class> indicates whether its a feasibility problem (\'f\', optimal value is zero) or not (\'o\')",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p','--prefix', type=str, default='.',
                        help="Folder to put these test problem definitions")
    parser.add_argument('-t','--time_horizons', nargs='+', type=float,
                        help="Time horizons of problems to generate")
    args = parser.parse_args()
    if args.time_horizons is None:
        parser.print_help()
        return
    if not os.path.isdir(args.prefix):
        os.mkdir(args.prefix)
    gen(args.prefix, args.time_horizons)

if __name__ == "__main__":
    main()