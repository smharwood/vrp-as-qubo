# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:05:23 2018

@author: smharwo
"""

"""
Simple example from
Desrochers, Desrosiers, Solomon, "A new optimization algorithm for the vehicle routing problem with time windows"
to test stuff
"""
import numpy as np
import sequence_based.RoutingProblem as rp

# NOTE
# With only one vehicle, this problem will be infeasible given the way time windows are handled:
# To be conservative about timing, we assume vehicles leave at the very end of the time window
# Need at least two vehicles
def DefineProblem(numVehicles, maxSequenceLength):
    """ Define a simple problem """

    # Get object,
    # number of vessels/vehicles,
    # Maximum number of visits in a sequence
    # D-1-2-3-D
    # Alternatively, if we cared more about capacity, 
    # we would link max number of stops with vehicle capacity and demand
    prob = rp.RoutingProblem()
    prob.setMaxVehicles(numVehicles)
    prob.setMaxSequenceLength(maxSequenceLength)
    
    # A depot node is required
    prob.addDepot('D',(0,np.inf))

    # Add nodes (Name, Time Window)
    # Nodes have the same level of demand
    prob.addNode('1',(1,7))
    prob.addNode('2',(2,4))
    prob.addNode('3',(4,7))

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

def getQUBO(numVehicles=2, maxSequenceLength=4, feasibility=False):
    """
    Define the problem and actually get back the QUBO matrix
    """
    prob = DefineProblem(numVehicles, maxSequenceLength)
    # get matrix and constant defining QUBO
    # use automatically calculated penalty parameter
    return prob.getQUBO(None,feasibility)

def getCplexProb(numVehicles=2, maxSequenceLength=4):
    """
    Define the problem and get CPLEX object encoding problem
    """
    prob = DefineProblem(numVehicles, maxSequenceLength)
    return prob.getCplexProb()


def test(feas):
    NV = 2
    SL = 4
    prob = DefineProblem(NV, SL)
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
    return

def test_solve():
    NV = 2
    SL = 4
    prob = DefineProblem(NV, SL)
    cp = prob.getCplexProb()
    print('Num variables: {}'.format(prob.getNumVariables()))
    prob.export_mip('ExSmall.lp')
    cp.solve()
    soln = cp.solution.get_values()
    print("Solution value", cp.solution.get_objective_value())
    routes = prob.getRoutes(soln)
    print("\nSolution status: "+cp.solution.get_status_string())
    print("Routes (Node sequences):")
    for r in routes: print(r)
    return

    
if __name__ == "__main__":
    # test(feas=True)
    test_solve()
