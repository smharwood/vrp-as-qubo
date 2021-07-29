# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:37:45 2019

@author: smharwo

Test instances for Stanford/Cornell:
vary TimeHorizon and number of routes added to get different size problems
"""

import argparse, os
from QUBOTools import QUBOContainer
import ExMIRPg1 as ex

def main(prefix, horizons=None):
    if horizons is None:
        horizons = [20]
    for TH in horizons:
        # define problem and export it
        prob = ex.DefineProblem(TH)
        
        print('Number of variables/routes: {}'.format(prob.getNumVariables()))
        sizing  = "_{}_{}_".format(len(prob.Nodes)-1, prob.getNumVariables())
        ising_f_name = os.path.join(prefix, "test"+sizing+"f.rudy")
        ising_o_name = os.path.join(prefix, "test"+sizing+"o.rudy")

        # Version with objective:
        Q, c = prob.getQUBO(None, feasibility=False)
        QC = QUBOContainer(Q, c)
        QC.export(ising_o_name, as_ising=True)
        # Feasibility version
        Q, c = prob.getQUBO(None, feasibility=True)
        QC = QUBOContainer(Q, c)
        QC.export(ising_f_name, as_ising=True)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        "Build a set of test problems\n"+
        "For example, run as\n"+
        "python generate_test_set.py -p TestSet -t 20 30 40 50\n"+
        "to get a test set with four instances of various sizes",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p','--prefix', type=str, default='.',
                        help="Folder to put these test problem definitions")
    parser.add_argument('-t','--time_horizons', nargs='+', type=float,
                        help="Time horizons of problems to generate")
    args = parser.parse_args()
    if not os.path.isdir(args.prefix):
        os.mkdir(args.prefix)
    main(args.prefix, args.time_horizons)
