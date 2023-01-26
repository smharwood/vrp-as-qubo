# -*- coding: utf-8 -*-
"""
Created on 5 December 2019

@author: stuart.m.harwood@exxonmobil.com
"""
import time
import numpy as np
import arc_based.RoutingProblem as rp

"""
Adapting something that looks like a MIRPLib instance to fit 
a sequence-based Vehicle Routing Problem with Time Windows formulation
Specifically, looking at LR1_DR02_VC01_V6a from
https://mirplib.scl.gatech.edu/sites/default/files/LR1_DR02_VC01_V6a.txt
This is one of the first "group 2" instances

Six identical vessels, one supply port, two demand ports
Time horizon can be adjusted
"""

def dTimeWindow(prevVisits,initial,rate,size,tankage):
    # inventory = initial + t*rate
    # Earliest a ship can discharge a full load into inventory:
    # inventory + (prevVisits+1)*size <= tankage
    tw0 = (tankage - (prevVisits+1)*size - initial)/rate
    # latest a ship can arrive before port runs out of inventory:
    # inventory + (prevVisits)*size < 0                
    tw1 = (-(prevVisits)*size - initial)/rate
    return (tw0,tw1)
  
def addDemandNodes(problem,Name,initial,rate,size,tankage, TimeHorizon):
    # add demand nodes for this given port
    prevVisits = 0
    DemandList = []
    while True:
        TW = dTimeWindow(prevVisits,initial,rate,size,tankage)
        if TW[1] > TimeHorizon:
            break
        # otherwise the time window is within the time horizon
        DemandList.append('{}-{}'.format(Name,prevVisits))
        problem.addNode(DemandList[-1],TW)
        prevVisits+=1
    return DemandList

def sTimeWindow(prevVisits,initial,rate,size,tankage):
    # inventory = initial + t*rate
    # Earliest a ship can load a full shipload:
    # inventory - (prevVisits+1)*size >= 0     
    tw0 = ((prevVisits+1)*size - initial)/rate
    # latest a ship can arrive before port capacity is exceeded:
    # inventory - (prevVisits)*size > tankage                
    tw1 = (tankage + (prevVisits)*size - initial)/rate
    return (tw0,tw1)
  
def addSupplyNodes(problem,SupplyName,initial,rate,size,tankage, TimeHorizon):
    prevVisits = 0
    SupplyList = []
    while True:
        TW = sTimeWindow(prevVisits,initial,rate,size,tankage)
        if TW[1] > TimeHorizon:
            break
        # otherwise the time window is within the time horizon
        SupplyList.append('{}-{}'.format(SupplyName,prevVisits))
        problem.addNode(SupplyList[-1],TW)
        prevVisits+=1
    return SupplyList

def DefineProblem(TimeHorizon):
    """
    Define a specific problem given a time horizon
    """
    # Create a routing problem
    prob = rp.RoutingProblem()
    
    # CargoSize is actual vehicle capacity
    CargoSize = 300
    
    # A depot node is required
    prob.addDepot('Depot',(0,np.inf))
    
    # Define demand node data
    Names =             ['D1', 'D2']
    initInventory =     [192,   220]
    consumptionRate =   [-32,   -40]
    tankage =           [384,   440]
    
    # Add demand nodes to problem
    DemandLists = []
    for (name,ini,rate,tank) in zip(Names,initInventory,consumptionRate,tankage):
        DemandLists.append(addDemandNodes(prob,name,ini,rate,CargoSize,tank,TimeHorizon))

    # Define supply node data
    Names =             ['S']
    initInventory =     [216]
    productionRate =    [72]
    tankage =           [432]
    
    # Add supply nodes to problem
    SupplyLists = []
    for (name,ini,rate,tank) in zip(Names,initInventory,productionRate,tankage):
        SupplyLists.append(addSupplyNodes(prob,name,ini,rate,CargoSize,tank,TimeHorizon))

    # Time points
    # This is tricky; we want as much resolution as possible,
    # but keep it as small as possible
    # Here, note that the travel times are all integer-valued
    # So, let time points be the integers that fall in any node's time window
    tw_points = []
    for n in prob.Nodes:
        if n.getName() == 'Depot':
            # depot's window is infinite
            continue
        (tw0, tw1) = n.getWindow()
        tw_points += np.arange(np.ceil(tw0),np.floor(tw1)+1).tolist()
    tw_points.append(0)
    tw_points = set(tw_points) # get unique values
    timepoints = list(tw_points)
    timepoints.sort()
    prob.addTimePoints(timepoints)

    # Arcs
    # Regular supply/demand nodes are fully connected,
    # have simple travel times based on the location
    # (and including any loading/unloading times)
    # Costs include port fees, etc, but note they are not symmetric
    # (because the nodes have a time component, not all arcs are physically reasonable-
    # but RoutingProblem checks for that)
    for s in SupplyLists[0]:
        for d in DemandLists[0]:
            prob.addArc(s,d, 8, 475)
            prob.addArc(d,s, 8, 511)
    for s in SupplyLists[0]:
        for d in DemandLists[1]:
            prob.addArc(s,d, 14, 880)
            prob.addArc(d,s, 14, 908)
    ## We are not allowing split deliveries, so
    ## travel directly between demand ports will never happen
    
    # Entry arcs (from Depot) enforce initial conditions of vessels
    # There are a few ways to do this, and may require dummy nodes, 
    # but for this problem, we just define arcs from depot to appropriate nodes
    # with travel times equal to the start time of the vessel (but zero cost)
    # vessel 4 (starts at S at t=2)
    entry_cost = 0
    prob.addArc('Depot','S-0',time=2,  cost=entry_cost)
    # vessel 1 (starts at S at t=5)
    prob.addArc('Depot','S-1',time=5,  cost=entry_cost)
    # vessel 3 (starts at S at t=8)
    prob.addArc('Depot','S-2',time=8,  cost=entry_cost)
    # vessel 0 (starts at D2 at t=0)
    prob.addArc('Depot','D2-0',time=0, cost=entry_cost)
    # vessel 2 (starts at D2 at t=10)
    prob.addArc('Depot','D2-1',time=10,cost=entry_cost)
    # vessel 5 (starts at D1 at t=4)
    prob.addArc('Depot','D1-0',time=4, cost=entry_cost)
    
    # Exiting arcs (back to Depot)
    # For simplicity, allow exit from any "regular" supply/demand node
    # No time nor cost
    for s in SupplyLists[0]:
        prob.addArc(s,'Depot', 0, 0)
    for d in DemandLists[0]:
        prob.addArc(d,'Depot', 0, 0)
    for d in DemandLists[1]:
        prob.addArc(d,'Depot', 0, 0)

    return prob

def getQUBO(TimeHorizon, feasibility=False):
    """
    Define the problem and actually get back the QUBO matrix
    """
    prob = DefineProblem(TimeHorizon)
    # get matrix and constant defining QUBO
    # use automatically calculated penalty parameter
    return prob.getQUBO(None,feasibility)

def getCplexProb(TimeHorizon):
    """
    Define the problem and get CPLEX object encoding problem
    """
    prob = DefineProblem(TimeHorizon)
    return prob.getCplexProb()

def test(feas):
    if feas:
        print("\nRunning as FEASIBILITY problem")
    else:
        print("\nRunning as normal optimality problem")

    TH = 100
    start = time.time()
    prob = DefineProblem(TH)
    print("Problem defined, exporting")
    print("time so far: {}".format(time.time() - start))
    prob.export_mip('ExMIRPg2.lp')
    print("time so far: {}".format(time.time() - start))

    # How big is it
    print("Time points: {}".format(prob.TimePoints))
    print('Nodes')
    for n in prob.Nodes:
        print(n)
    #print('Arcs')
    #for a in prob.Arcs.values():
    #    print(a)
    print('Num variables: {}'.format(prob.getNumVariables()))
    Q, c = prob.getQUBO(None,feas)
    print("time so far: {}".format(time.time() - start))
    shape = Q.shape
    nnz = Q.nnz
    print('QUBO: {}x{} w/ {} nonzeros'.format(shape[0],shape[1],nnz))


def test_with_solve(feas, enum_opt=False, enum_feas=False):
    if feas:
        print("\nRunning as FEASIBILITY problem")
    else:
        print("\nRunning as normal optimality problem")

    TH = 20
    prob = DefineProblem(TH)
    sol_time = time.time()
    prob.solveCplexProb('ExMIRPg2.sol')
    sol_time = time.time() - sol_time

    # How big is it
    print("Time points: {}".format(prob.TimePoints))
    print('Nodes')
    for n in prob.Nodes:
        print(n)
    print('Arcs')
    for a in prob.Arcs.values():
        print(a)
    print('Num variables: {}'.format(prob.getNumVariables()))
    Q, c = prob.getQUBO(None,feas)
    shape = Q.shape
    nnz = Q.nnz
    print('QUBO: {}x{} w/ {} nonzeros'.format(shape[0],shape[1],nnz))

    n_opt_sol = 0
    n_feas_sol = 0
    if enum_opt:
        n_opt_sol = count_sol(prob.getCplexProb(), criteria=0)
    if enum_feas:
        n_feas_sol = count_sol(prob.getCplexProb(), criteria=1)
    export_sol_metrics(sol_time, n_opt_sol, n_feas_sol)

def test_print():
    prob = DefineProblem(31)
    cp = prob.getCplexProb()
    print('Num variables: {}'.format(prob.getNumVariables()))
    prob.export_mip('ExMIRPg2.lp')
    cp.solve()
    soln = cp.solution.get_values()
    routes = prob.getRoutes(soln)
    print("\nSolution status: "+cp.solution.get_status_string())
    print("Routes (Node,time sequences):")
    for r in routes: print(r)
    return

if __name__ == "__main__":
    #test(feas=False)
    #test_with_solve(feas=True, enum_opt=True, enum_feas=False)
    test_print()
