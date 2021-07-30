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
from path_based.RoutingProblem import RoutingProblem as PathBasedRoutingProblem
#from optimization.mirp_encoding import QUBOTools as qt

def DefineProblem():
    prob = PathBasedRoutingProblem()

    # Set vehicle capacity
    # Vehicles leave depot fully loaded
    prob.setVehicleCap(6)
    prob.setInitialLoading(6)

    # Add nodes (Name, Demand, Time Window=(0,infty))
    prob.addNode('D', 0)
    prob.addNode('1', 1, (1,7))
    prob.addNode('2', 2, (2,4))
    prob.addNode('3', 2, (4,7))
    prob.setDepot('D')

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


    # Add/check routes
    # From paper, we know there are 11
    R = [None]*13
    R[0] = ['D','1','D']
    R[1] = ['D','2','D']
    R[2] = ['D','3','D']
    R[3] = ['D','1','2','D']
    R[4] = ['D','1','3','D']
    R[5] = ['D','2','1','D']
    R[6] = ['D','2','3','D']
    R[7] = ['D','3','1','D']
    R[8] = ['D','1','2','3','D']
    R[9] = ['D','2','1','3','D']
    R[10]= ['D','2','3','1','D']
    # not valid routes, to test:
    R[11]= ['D','3','1','2','D']     # timing not right
    R[12]= ['D','2','3','1','3','D'] # doubles up a node

    for route in R:
        f, _, _ = prob.addRoute(route)
        if not f:
            print(str(route)+' not feasible')
    return prob

def getQUBO(feasibility=False):
    """
    Define the problem and actually get back the QUBO matrix
    """
    prob = DefineProblem()
    # get matrix and constant defining QUBO
    # use automatically calculated penalty parameter
    return prob.getQUBO(None,feasibility)

def getCplexProb():
    """
    Define the problem and get CPLEX object encoding problem
    """
    prob = DefineProblem()
    return prob.getCplexProb()


def test(feas):
    prob = DefineProblem()
    blec_cost, blec_constraints_matrix, blec_constraints_rhs = prob.getBLECdata()
    Q,c = prob.getQUBO(feasibility=feas)

    # Test a feasible solution
    x = np.zeros(11)
    x[10] = 1
    res = np.linalg.norm(blec_constraints_matrix.dot(x) - blec_constraints_rhs)
    assert np.isclose(res,0.0), 'Feasible solution does not appear feasible'
    # how does objective compare to QUBO?
    # i know x is feasible, so ignore res
    mip_obj = 0
    mip_obj += 0 if feas else blec_cost.dot(x)
    qubo_obj = c + Q.dot(x).dot(x)
    assert np.isclose(mip_obj,qubo_obj), 'QUBO and MIP do not agree'

    # another feasible
    x = np.zeros(11)
    x[0] = 1
    x[1] = 1
    x[2] = 1
    res = np.linalg.norm(blec_constraints_matrix.dot(x) - blec_constraints_rhs)
    assert np.isclose(res,0.0), 'Feasible solution does not appear feasible'
    # how does objective compare to QUBO?
    mip_obj = 0
    mip_obj += 0 if feas else blec_cost.dot(x)
    qubo_obj = c + Q.dot(x).dot(x)
    assert np.isclose(mip_obj,qubo_obj), 'QUBO and MIP do not agree'

    # an INfeasible solution
    x = np.zeros(11)
    x[0] = 1
    x[1] = 1
    x[2] = 1
    x[3] = 1
    res = np.linalg.norm(blec_constraints_matrix.dot(x) - blec_constraints_rhs)
    assert res > 0.0, 'INfeasible solution appears feasible'
    if feas:
        qubo_obj = c + Q.dot(x).dot(x)
        mip_penalty = res**2
        assert np.isclose(mip_penalty,qubo_obj), 'QUBO and MIP do not agree'
    return

def test_print():
    prob = DefineProblem()
    cp = prob.getCplexProb()
    cp.solve()
    soln = cp.solution.get_values()
    routes = prob.getRoutes(soln)
    print("\nSolution status: "+cp.solution.get_status_string())
    print("Routes (Node sequences):")
    for r in routes: print(r)
    return

def test_export():
    print('\n### Testing export functions ###')
    feas = False
    prob = DefineProblem()
    Q,c = prob.getQUBO(feasibility=feas)
    qubo = qt.QUBOContainer(Q,c)
    qubo.export('small.qubo')
    prob.exportQUBO('small_og.qubo',feas)
    prob.export_mip('ExSmall.lp')
    return


if __name__ == "__main__":
    test(feas=False)
    test(feas=True)
    test_print()
    test_export()
