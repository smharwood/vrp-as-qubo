# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:37:45 2019

@author: smharwo

Test instances for Stanford/Cornell:
vary TimeHorizon and number of routes added to get different size problems
"""
import argparse, os
from itertools import product
from QUBOTools import QUBOContainer
import arc_based.ExMIRPg1 as abex
import path_based.ExMIRPg1 as pbex
import sequence_based.ExMIRPg1 as sbex


def main(prefix, horizons=None):
    if horizons is None:
        horizons = [20]
    
    formulations = [abex, pbex, sbex]
    for (TH, ex) in product(horizons, formulations):
        mod = ex.__name__.split('.')[0]
        name = ''.join([w[0] for w in mod.split('_')])

        # define problem and export it
        prob = ex.DefineProblem(TH)
        
        # Version with objective:
        Q, c = prob.getQUBO(None, feasibility=False)
        QC = QUBOContainer(Q, c)
        print("Time horizon {}, formulation {}, number of variables: {}".
            format(TH, mod, prob.getNumVariables()))
        bname = "test_{}_{}_".format(name, prob.getNumVariables())
        ising_o_name = os.path.join(prefix, bname + "o.rudy")
        QC.export(ising_o_name, as_ising=True)

        # Feasibility version
        Q, c = prob.getQUBO(None, feasibility=True)
        QC = QUBOContainer(Q, c)
        ising_f_name = os.path.join(prefix, bname + "f.rudy")
        QC.export(ising_f_name, as_ising=True)
    return

if __name__ == "__main__":
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
    if not os.path.isdir(args.prefix):
        os.mkdir(args.prefix)
    main(args.prefix, args.time_horizons)
