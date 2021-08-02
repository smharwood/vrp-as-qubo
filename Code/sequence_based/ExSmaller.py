# -*- coding: utf-8 -*-
"""
Created on 23 December 2019

@author: smharwo
"""
import time
import numpy as np
from sequence_based.RoutingProblem import RoutingProblem as SequenceBasedRoutingProblem


def DefineProblem(numVehicles=1):
    """ Define a simple problem """

    # Get object,
    # number of vessels/vehicles,
    # Maximum number of visits in a sequence
    # D-1-2-3-D
    # Alternatively, if we cared more about capacity, 
    # we would link max number of stops with vehicle capacity and demand
    prob = SequenceBasedRoutingProblem()
    prob.setMaxVehicles(numVehicles)
    prob.setMaxSequenceLength(6) # make this larger than necessary (5) to test "absorbing" depot
    
    # A depot node is required
    prob.addDepot('D',(0,np.inf))

    # Add nodes (Name, Time Window)
    # Nodes have the same level of demand
    prob.addNode('1',(1,1))
    prob.addNode('2',(2,2))
    prob.addNode('3',(3,3))
    
    # Add arcs (Origin, Destination, Time, Cost=0)
    # "enter"
    prob.addArc('D','1',1)
    prob.addArc('D','2',2)
    prob.addArc('D','3',3)

    # "regular" arcs: very few
    prob.addArc('1','2',1)
    prob.addArc('2','3',1)

    # "exit"
    prob.addArc('1','D',0)
    prob.addArc('2','D',0)
    prob.addArc('3','D',0)

    return prob

def getQUBO(numVehicles, feasibility=False):
    """
    Define the problem and actually get back the QUBO matrix
    """
    prob = DefineProblem(numVehicles)
    # get matrix and constant defining QUBO
    # use automatically calculated penalty parameter
    return prob.getQUBO(None,feasibility)

def getCplexProb(numVehicles):
    """
    Define the problem and get CPLEX object encoding problem
    """
    prob = DefineProblem(numVehicles)
    return prob.getCplexProb()


def test():
    NV = 1
    feas = True
    prob = DefineProblem(NV)
    Q, c = prob.getQUBO(None,feas)

    # How big is it
    print('Nodes')
    for n in prob.Nodes:
        print(n)
    print('Arcs')
    for a in prob.Arcs.values():
        print(a)
    print('Num variables: {}'.format(prob.getNumVariables()))
    
    # QUBO
    shape = Q.shape
    nnz = Q.nnz
    print("QUBO constant: {}".format(c))
    print('QUBO: {}x{} w/ {} nonzeros'.format(shape[0],shape[1],nnz))
    print("QUBO mat:")
    prob.print_qubo(Q)
    return

def test_with_solve(enum_opt=False, enum_feas=False):
    NV = 1
    feas = True
    prob = DefineProblem(NV)
    Q, c = prob.getQUBO(None,feas)

    sol_time = time.time()
    prob.solveCplexProb('ExSmallest.sol')
    sol_time = time.time() - sol_time

    # n_opt_sol = 0
    # n_feas_sol = 0
    # if enum_opt:
    #     n_opt_sol = count_sol(prob.getCplexProb(), criteria=0)
    # if enum_feas:
    #     n_feas_sol = count_sol(prob.getCplexProb(), criteria=1)
    #
    # export_sol_metrics(sol_time, n_opt_sol, n_feas_sol)


    
if __name__ == "__main__":
    test()
    test_with_solve()
