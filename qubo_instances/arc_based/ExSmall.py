# -*- coding: utf-8 -*-
"""
Created on 6 January 2020

@author: smharwo
"""

"""
Simple example from
Desrochers, Desrosiers, Solomon, "A new optimization algorithm for the vehicle routing problem with time windows"
to test stuff
"""
import numpy as np
import arc_based.RoutingProblem as rp
#import optimization.mirp_encoding.QUBOTools as qt


def DefineProblem(time_points=None):
    """ Define a simple problem

    args:
    time_points (list): The set of discrete time points for the problem

    return:
    prob (arc_based.RoutingProblem): An arc-based RoutingProblem object representing the problem
    """

    if time_points is None:
        time_points = [0,1,2,4,7]

    # Get object
    prob = rp.RoutingProblem()
    
    # A depot node is required
    prob.addDepot('D',(0,np.inf))

    # Add nodes (Name, Time Window)
    # Nodes have the same level of demand
    prob.addNode('1',(1,7))
    prob.addNode('2',(2,4))
    prob.addNode('3',(4,7))

    # add time points
    #prob.addTimePoints(list(range(0,8)))
    prob.addTimePoints(time_points)

    # Add arcs (Origin, Destination, Time, Cost=0)
    prob.addArc('D','1', 1, 1)
    prob.addArc('D','2', 2, 2)
    prob.addArc('D','3', 2, 2)

    prob.addArc('1','D', 1, 1)
    prob.addArc('1','2', 1, 1)
    prob.addArc('1','3', 1, 1)

    prob.addArc('2','D', 2, 2)
    prob.addArc('2','1', 1, 1)
    prob.addArc('2','3', 1, 1)

    prob.addArc('3','D', 2, 2)
    prob.addArc('3','1', 1, 1)

    return prob

def getQUBO(feasibility=False):
    """
    Define the problem and actually get back the QUBO matrix
    """
    prob = DefineProblem()
    # get matrix and constant defining QUBO
    # use automatically calculated penalty parameter
    return prob.getQUBO(None, feasibility)

def getCplexProb():
    """
    Define the problem and get CPLEX object encoding problem
    """
    prob = DefineProblem()
    return prob.getCplexProb()


def test(feas):

    if feas:
        print("\nRunning as FEASIBILITY problem")
    else:
        print("\nRunning as normal optimality problem")
    prob = DefineProblem()
    prob.build_blec_constraints()
    prob.build_blec_obj()
    Q, c = prob.getQUBO(None,feas)
    prob.export_mip('ExSmall.lp')

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

    # QUBO object to calculate objectives
    qubo = qt.QUBOContainer(Q,c,'symmetric')

    # Binary variable solution:
    # We expect D-1-2-3-D to be feasible (if we get timing right)
    # This uses arcs with cost 0; expect objective = 0
    x = np.zeros(prob.getNumVariables())
    # D - 1
    i = prob.getNodeIndex('D')
    s = 0
    j = prob.getNodeIndex('1')
    t = 1
    x[prob.getVarIndex(i,s,j,t)] = 1
    # 1 - 2
    i = prob.getNodeIndex('1')
    s = 1
    j = prob.getNodeIndex('2')
    t = 2
    x[prob.getVarIndex(i,s,j,t)] = 1
    # 2 - 3
    i = prob.getNodeIndex('2')
    s = 2
    j = prob.getNodeIndex('3')
    t = 4
    x[prob.getVarIndex(i,s,j,t)] = 1
    # 3 - D
    i = prob.getNodeIndex('3')
    s = 4
    j = prob.getNodeIndex('D')
    t = 7
    x[prob.getVarIndex(i,s,j,t)] = 1

    assert sum(x) == 4, 'Some arc not allowed'
    print("\nFeasible configuration?: {}".format(x))
    print("Objective value: {}".format(qubo.evaluate_QUBO(x)))

    # Another:
    # We expect D-1-D and D-2-3-D to be feasible (if we get timing right)
    # This uses the expensive arc D-2
    x = np.zeros(prob.getNumVariables())
    # D - 1
    i = prob.getNodeIndex('D')
    s = 0
    j = prob.getNodeIndex('1')
    t = 1
    x[prob.getVarIndex(i,s,j,t)] = 1
    # 1 - D
    i = prob.getNodeIndex('1')
    s = 1
    j = prob.getNodeIndex('D')
    t = 2
    x[prob.getVarIndex(i,s,j,t)] = 1

    # D - 2
    i = prob.getNodeIndex('D')
    s = 0
    j = prob.getNodeIndex('2')
    t = 2
    x[prob.getVarIndex(i,s,j,t)] = 1
    # 2 - 3
    i = prob.getNodeIndex('2')
    s = 2
    j = prob.getNodeIndex('3')
    t = 4
    x[prob.getVarIndex(i,s,j,t)] = 1
    # 3 - D
    i = prob.getNodeIndex('3')
    s = 4
    j = prob.getNodeIndex('D')
    t = 7
    x[prob.getVarIndex(i,s,j,t)] = 1

    assert sum(x) == 5, 'Some arc not allowed'
    print("\nFeasible configuration?: {}".format(x))
    print("Objective value: {}".format(qubo.evaluate_QUBO(x)))

    # Test INfeasible solution
    # We expect D-1-2-D to be INfeasible (not visiting 3)
    x = np.zeros(prob.getNumVariables())
    # D - 1
    i = prob.getNodeIndex('D')
    s = 0
    j = prob.getNodeIndex('1')
    t = 1
    x[prob.getVarIndex(i,s,j,t)] = 1
    # 1 - 2
    i = prob.getNodeIndex('1')
    s = 1
    j = prob.getNodeIndex('2')
    t = 2
    x[prob.getVarIndex(i,s,j,t)] = 1
    # 2 - D
    i = prob.getNodeIndex('2')
    s = 2
    j = prob.getNodeIndex('D')
    t = 4
    x[prob.getVarIndex(i,s,j,t)] = 1
    
    assert sum(x) == 3, 'Some arc not allowed'
    print("\nINFeasible configuration?: {}".format(x))
    print("Objective value: {}".format(qubo.evaluate_QUBO(x)))
    return

def test_print():
    prob = DefineProblem()
    cp = prob.getCplexProb()
    cp.solve()
    soln = cp.solution.get_values()
    routes = prob.getRoutes(soln)
    print("\nSolution status: "+cp.solution.get_status_string())
    print("Routes (Node,time sequences):")
    for r in routes: print(r)
    return

if __name__ == "__main__":
#    test(feas=True)
#    test(feas=False)
    test_print()
