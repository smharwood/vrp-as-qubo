# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:38:19 2018

@author: stuart.m.harwood@exxonmobil.com
"""
"""
Adapting something that looks like a MIRPLib instance to fit the VRPTW formalism
Specifically, looking at LR1_DR02_VC01_V6a from
https://mirplib.scl.gatech.edu/sites/default/files/LR1_DR02_VC01_V6a.txt
This is one of the first "group 2" instances

Six identical vessels, one supply port, two demand ports
Time horizon can be adjusted
"""

import RoutingProblem as rp


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
        problem.addNode(DemandList[-1],size,TW)
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
        problem.addNode(SupplyList[-1],-size,TW)
        prevVisits+=1
    return SupplyList

def DefineProblem(TimeHorizon):
    """
    Define a specific problem given a time horizon
    """
    # Create a routing problem
    prob = rp.RoutingProblem()
    
    # CargoSize is actual vehicle capacity,
    # but in RoutingProblem set it to ZERO
    # The trick we will employ is to give supply regions "negative" demand,
    # so that the overall running total of demand met for a valid route will always be nonpositive
    prob.setVehicleCap(0)
    CargoSize = 300
    
    # A depot node is required (zero demand)
    prob.addNode('Depot',0)
    prob.setDepot('Depot')
    
    # Define demand node data
    Names =             ['D1',  'D2']
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
    
    # Dummy Supply nodes help control where vessels are initially available
    prob.addNode('Dum1',-CargoSize)
    prob.addNode('Dum2',-CargoSize)
    prob.addNode('Dum3',-CargoSize)
    
    # Arcs
    # Regular supply/demand nodes are fully connected,
    # have simple travel times based on the location
    # (and including any loading/unloading times)
    # Costs include port fees, etc, but note they are not symmetric
    # (because the nodes have a time component, not all arcs are physically reasonable-
    # but checking for a feasible/valid route will catch that)
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
    
    # TODO:
    # all these regular arcs have positive cost-
    # there is no inherent incentive (in the objective) to keep making trips
    # This may make generating routes by a traditional dynamic programming approach a little tricky

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
    prob.addArc('Depot','Dum1',0)
    prob.addArc('Dum1','D2-0',0,0)
    # vessel 2:
    prob.addArc('Depot','Dum2',0)
    prob.addArc('Dum2','D2-1',10,0)
    # vessel 5:
    prob.addArc('Depot','Dum3',0)
    prob.addArc('Dum3','D1-0',4,0)
    
    # Exiting arcs (back to Depot)
    # For simplicity, allow exit from any "regular" supply/demand node
    # No time, but at a high cost, to discourage premature exit
    # Cost should be approx as expensive as a "full" route, visiting all nodes
    #   (for this problem, supply node must be visited every 6 days- 
    #    this gives indicator of how expensive a full route is)
    # TODO:
    #   Reconsider?
    for s in SupplyLists[0]:
        prob.addArc(s,'Depot', 0, (TimeHorizon/6.0)*1500)
    for d in DemandLists[0]:
        prob.addArc(d,'Depot', 0, (TimeHorizon/6.0)*1500)
    for d in DemandLists[1]:
        prob.addArc(d,'Depot', 0, (TimeHorizon/6.0)*1500)

    return prob