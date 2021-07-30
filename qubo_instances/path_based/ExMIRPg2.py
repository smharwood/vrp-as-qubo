# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:38:19 2018

@author: stuart.m.harwood@exxonmobil.com

Adapting something that looks like a MIRPLib instance to fit the VRPTW formalism
Specifically, looking at LR1_DR02_VC01_V6a from
https://mirplib.scl.gatech.edu/sites/default/files/LR1_DR02_VC01_V6a.txt
This is one of the first "group 2" instances

Six identical vessels, one supply port, two demand ports
Time horizon can be adjusted
"""
import numpy
import time, logging
from path_based.RoutingProblem import RoutingProblem as PathBasedRoutingProblem
#from optimization.mirp_encoding.utils import count_sol, export_sol_metrics


# helper functions
def dTimeWindow(prevVisits, initial, rate, size, tankage):
    # inventory = initial + t*rate
    # Earliest a ship can discharge a full load into inventory:
    # inventory + (prevVisits+1)*size <= tankage
    tw0 = (tankage - (prevVisits+1)*size - initial)/rate
    # latest a ship can arrive before port runs out of inventory:
    # inventory + (prevVisits)*size < 0                
    tw1 = (-(prevVisits)*size - initial)/rate
    return (tw0,tw1)
  
def addDemandNodes(problem, Name, initial, rate, size, tankage, TimeHorizon):
    """ Add nodes corresponding to this demand port """
    prevVisits = 0
    DemandList = []
    while True:
        TW = dTimeWindow(prevVisits, initial, rate, size, tankage)
        if TW[1] > TimeHorizon:
            break
        # otherwise the time window is within the time horizon
        DemandList.append('{}-{}'.format(Name, prevVisits))
        problem.addNode(DemandList[-1], size, TW)
        prevVisits+=1
    return DemandList

def sTimeWindow(prevVisits, initial, rate, size, tankage):
    # inventory = initial + t*rate
    # Earliest a ship can load a full shipload:
    # inventory - (prevVisits+1)*size >= 0     
    tw0 = ((prevVisits+1)*size - initial)/rate
    # latest a ship can arrive before port capacity is exceeded:
    # inventory - (prevVisits)*size > tankage                
    tw1 = (tankage + (prevVisits)*size - initial)/rate
    return (tw0,tw1)
  
def addSupplyNodes(problem, SupplyName, initial, rate, size, tankage, TimeHorizon):
    """ Add nodes corresponding to this supply port """
    prevVisits = 0
    SupplyList = []
    while True:
        TW = sTimeWindow(prevVisits, initial, rate, size, tankage)
        if TW[1] > TimeHorizon:
            break
        # otherwise the time window is within the time horizon
        SupplyList.append('{}-{}'.format(SupplyName, prevVisits))
        problem.addNode(SupplyList[-1], -size, TW)
        prevVisits+=1
    return SupplyList

def DefineProblem(TimeHorizon):
    """
    Define a specific problem given a time horizon
    """
    # Create a routing problem
    prob = PathBasedRoutingProblem()

    # CargoSize is vessel capacity,
    # Depot node is a dummy node, so vessels are initially empty and must first pass through a 
    # supply node before they can deliver anything
    CargoSize = 300
    prob.setVehicleCap(CargoSize)
    prob.setInitialLoading(0)

    # A depot node is required (zero demand)
    prob.addNode('Depot',0)
    prob.setDepot('Depot')

    # Define demand node data
    Names =             ['D1',  'D2']
    initInventory =     [192,   220]
    consumptionRate =   [-32,   -40]
    tankage =           [384,   440]

    # Add demand nodes to problem
    # Demand nodes have postive demand level
    DemandPorts = []
    for (name,ini,rate,tank) in zip(Names,initInventory,consumptionRate,tankage):
        DemandPorts.append(addDemandNodes(prob,name,ini,rate,CargoSize,tank,TimeHorizon))

    # Define supply node data
    Names =             ['S']
    initInventory =     [216]
    productionRate =    [72]
    tankage =           [432]

    # Add supply nodes to problem
    # Supply nodes have negative demand level
    SupplyPorts = []
    for (name,ini,rate,tank) in zip(Names,initInventory,productionRate,tankage):
        SupplyPorts.append(addSupplyNodes(prob,name,ini,rate,CargoSize,tank,TimeHorizon))
        
    # Dummy Supply nodes help control where vessels are initially available
    prob.addNode('Dum1',-CargoSize)
    prob.addNode('Dum2',-CargoSize)
    prob.addNode('Dum3',-CargoSize)

    #print(prob.NameList)

    # Arcs
    # Regular supply/demand nodes are fully connected,
    # have simple travel times based on the location
    # (and including any loading/unloading times)
    # Costs include port fees, etc, but note they are not symmetric
    # (because the nodes have a time component, not all arcs are physically reasonable-
    # but checking for a feasible/valid route will catch that)
    for s in SupplyPorts[0]:
        for d in DemandPorts[0]:
            prob.addArc(s,d, 8, 475)
            prob.addArc(d,s, 8, 511)
    for s in SupplyPorts[0]:
        for d in DemandPorts[1]:
            prob.addArc(s,d, 14, 880)
            prob.addArc(d,s, 14, 908)
    ## We are not allowing split deliveries, so
    ## travel directly between demand ports will never happen

    # Entering/initial condition and exiting arcs

    # Special arcs to enforce initial conditions:
    # Vessels starting at supply ports
    # vessel 4:
    prob.addArc('Depot','S-0',2,0)
    # vessel 1:
    prob.addArc('Depot','S-1',5,0)
    # vessel 3:
    prob.addArc('Depot','S-2',8,0)
    # Vessels starting at demand ports, 
    # but which have initial loadings must go through a dummy port
    # vessel 0:
    prob.addArc('Depot','Dum1', 0)
    prob.addArc('Dum1','D2-0', 0, 0)
    # vessel 2:
    prob.addArc('Depot','Dum2', 0)
    prob.addArc('Dum2','D2-1', 10, 0)
    # vessel 5:
    prob.addArc('Depot','Dum3', 0)
    prob.addArc('Dum3','D1-0', 4, 0)

    # Exiting arcs (back to Depot)
    # For simplicity, allow exit from any "regular" supply/demand node
    final_cost = 0
    for s in SupplyPorts[0]:
        prob.addArc(s,'Depot', 0, final_cost)
    for d in DemandPorts[0]:
        prob.addArc(d,'Depot', 0, final_cost)
    for d in DemandPorts[1]:
        prob.addArc(d,'Depot', 0, final_cost)

    # Add routes/variables
    # high_cost should be approx as expensive as a "full" route, visiting all nodes
    #   (for this problem, supply node must be visited every 6 days- 
    #    this gives indicator of how expensive a full route is)
    high_cost = (TimeHorizon/6.0)*1500
    #add_variables(prob)
    add_routes(prob, TimeHorizon, high_cost)
    return prob

def add_routes(problem, TimeHorizon, high_cost):
    """ Add routes by a solution heuristic """
    # for reproducibility :
    numpy.random.seed(0)

    # Shortest routes possible
    problem.addRoute(['Depot', 'S-0', 'Depot'])
    problem.addRoute(['Depot', 'S-1', 'Depot'])
    problem.addRoute(['Depot', 'S-2', 'Depot'])
    problem.addRoute(['Depot', 'Dum1', 'D2-0', 'Depot'])
    problem.addRoute(['Depot', 'Dum2', 'D2-1', 'Depot'])
    problem.addRoute(['Depot', 'Dum3', 'D1-0', 'Depot'])

    # Make the depot high cost, to discourage premature exit, 
    # make initial nodes all equal high cost, because once a route visits it, no other route should,
    # make later arrival times more expensive, to favor early arrival,
    initial_nodes = ['S-0', 'S-1', 'S-2', 'Dum1', 'Dum2', 'Dum3', 'D1-0', 'D2-0', 'D2-1']
    time_costs = lambda t: 0 if t <=10 else 100*t
    node_costs = [0]*len(problem.Nodes)
    node_costs[problem.depotIndex] = high_cost
    for i_node in initial_nodes:
        node_costs[problem.getNodeIndex(i_node)] = high_cost

    for (explore, rep) in zip([0, 1, numpy.inf], [1, int(TimeHorizon), int(10*TimeHorizon)]):
        for _ in range(rep):
            problem.addRoutes_better(explore, node_costs, time_costs)
    return

def add_variables(problem):
    """
    DEPRECATED
    Add variables/routes to a problem in a consistent way
    """
    # for reproducibility :
    numpy.random.seed(0)

    problem.addRoutes(0.5, explore=0)
    problem.addRoutes(0.5, explore=numpy.inf)

    # Column generation at root node sort of works
#    start = time.time()
#    problem.addRoutes(0.9, explore=0)
#    problem.addRoutes_CG(add_rate=0.1, explore=numpy.inf)
#    print("Time: {}".format(time.time() - start))
    return

def route_print(problem, route):
    time = 0
    demand = 0
    prev_node = route[0]
    print("({}, {}) - ".format(problem.NodeNames[prev_node], time), end='')
    for node in route[1:]:
        arc = (prev_node, node)
        feasArc, time, demand = problem.checkArc(time, demand, arc)
        print("({:5}, {:5.2f}) - ".format(problem.NodeNames[node], time), end='')
        prev_node = node
    print()
    return

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


def test():
    # Test construction of QUBO
    TH = 18
    feas = False
    Q1,c1 = getQUBO(TH,feas)
    print("Constant: {}".format(c1))
    print("Q matrix:")
    (r,c) = Q1.shape
    for i in range(r):
        st = ""
        for j in range(c):
            st += '  {:12.2f}'.format(Q1[i,j])
        print(st)
    return

def test_with_solve(enum_opt=False, enum_feas=False):
    # Test construction of QUBO
    TH = 18
    feas = False
    Q1,c1 = getQUBO(TH,feas)

    prob = DefineProblem(TH)
    sol_time = time.time()
    prob.solveCplexProb('ExMIRPg2.sol')
    sol_time = time.time() - sol_time

    n_opt_sol = 0
    n_feas_sol = 0
    if enum_opt:
        n_opt_sol = count_sol(prob.getCplexProb(), criteria=0)
    if enum_feas:
        n_feas_sol = count_sol(prob.getCplexProb(), criteria=1)
    export_sol_metrics(sol_time, n_opt_sol, n_feas_sol)
    return

def test_print():
    prob = DefineProblem(15.4)
    cp = prob.getCplexProb()
    #for n in prob.Nodes: print(n)
    print('Num variables: {}'.format(prob.getNumVariables()))
    cp.write('ExMIRPg2.lp')
    cp.solve()
    soln = cp.solution.get_values()
    routes = prob.getRoutes(soln)
    print("\nSolution status: "+cp.solution.get_status_string())
    print("Routes (Node sequences):")
    for r in routes: 
        route_print(prob, r)
        #print(" - ".join(prob.getRouteNames(r)))
    return


if __name__ == "__main__":
    numpy.set_printoptions(precision=3)
    logging.basicConfig(filename='ExMirpg2.log', filemode='w', level=logging.INFO)
#    prob = DefineProblem(60)
#    prob.export_mip('ExMIRP.lp')
#    test()
#    test_with_solve(enum_opt=True, enum_feas=True)
    test_print()
