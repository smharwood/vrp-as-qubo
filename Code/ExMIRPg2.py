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

#%%
# Set up
import numpy
import RoutingProblem as rp

# Create a routing problem
prob = rp.RoutingProblem()

# CargoSize is actual vehicle capacity,
# but in RoutingProblem set it to ZERO
# The trick we will employ is to give supply regions "negative" demand,
# so that the overall running total of demand met for a valid route will always be nonpositive
prob.setVehicleCap(0)
CargoSize = 300
TimeHorizon = 40

# A depot node is required (zero demand)
prob.addNode('Depot',0)
prob.setDepot('Depot')

#%%
# Add demand nodes

# helper functions
def dTimeWindow(prevVisits,initial,rate,size,tankage):
    tw0 = None
    tw1 = None
    inventory = initial
    for t in range(TimeHorizon):
        # Earliest a ship can discharge a full load into inventory
        if inventory + (prevVisits+1)*size <= tankage and tw0 is None:
            tw0 = t
        # latest a ship can arrive before port runs out of inventory
        if inventory + (prevVisits)*size < 0          and tw1 is None:
            tw1 = t-1
        inventory+=rate
        if tw0 is not None and tw1 is not None:
            return (tw0,tw1)
    raise Exception
  
def addDemandNodes(problem,Name,initial,rate,size,tankage):
    # add demand nodes for this given port
    prevVisits = 0
    DemandList = []
    while True:
        try:
            TW = dTimeWindow(prevVisits,initial,rate,size,tankage)
            DemandList.append('{}-{}'.format(Name,prevVisits))
            problem.addNode(DemandList[-1],size,TW)
            prevVisits+=1
        except Exception as e:
            print(e)
            break
    return DemandList

# Define demand node data
Names =             ['D1',  'D2']
initInventory =     [192,   220]
consumptionRate =   [-32,   -40]
tankage =           [384,   440]

# Add demand nodes to problem
DemandLists = []
for (name,ini,rate,tank) in zip(Names,initInventory,consumptionRate,tankage):
    DemandLists.append(addDemandNodes(prob,name,ini,rate,CargoSize,tank))


#%%
# Add supply nodes

def sTimeWindow(prevVisits,initial,rate,size,tankage):
    tw0 = None
    tw1 = None
    inventory = initial
    for t in range(TimeHorizon):
        # Earliest a ship can load a full shipload
        if inventory - (prevVisits+1)*size >= 0     and tw0 is None:
            tw0 = t
        # latest a ship can arrive before port capacity is exceeded
        if inventory - (prevVisits)*size > tankage  and tw1 is None:
            tw1 = t-1
        inventory+=rate
        if tw0 is not None and tw1 is not None:
            return (tw0,tw1)
    raise Exception
  
def addSupplyNodes(problem,SupplyName,initial,rate,size,tankage):
    prevVisits = 0
    SupplyList = []
    while True:
        try:
            TW = sTimeWindow(prevVisits,initial,rate,size,tankage)
            SupplyList.append('{}-{}'.format(SupplyName,prevVisits))
            problem.addNode(SupplyList[-1],-size,TW)
            prevVisits+=1
        except Exception as e:
            print(e)
            break
    return SupplyList

# Define supply node data
Names =             ['S']
initInventory =     [216]
productionRate =    [72]
tankage =           [432]

# Add supply nodes to problem
SupplyLists = []
for (name,ini,rate,tank) in zip(Names,initInventory,productionRate,tankage):
    SupplyLists.append(addSupplyNodes(prob,name,ini,rate,CargoSize,tank))
    
#%%

# Dummy Supply nodes help control where vessels are initially available
prob.addNode('Dum1',-CargoSize)
prob.addNode('Dum2',-CargoSize)
prob.addNode('Dum3',-CargoSize)

print(prob.NameList)

#%%
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
#%%
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

#%%
# Add routes
# This will automatically generate up to the given number of routes
numpy.random.seed(0) # for reproducibility
# Play with route generation; try one or the other or both
vf = prob.addRoutes(50,explore=10)
vf = prob.addRoutes(50,explore=1,vf=vf)
vf = prob.addRoutes(50,explore=0,vf=vf)
prob.addRoutesFeasibility(50,explore=10) 
prob.addRoutesFeasibility(50,explore=1) 
prob.addRoutesFeasibility(50,explore=0) 


print('Number of variables/routes: {}'.format(len(prob.mip_variables)))

#%%

# Special Test instances for Stanford:
# vary TimeHorizon and number of routes added to get different size problems

sizing  = "_{}_{}_".format(len(prob.Nodes)-1, len(prob.mip_variables))
ising_f_name = "test"+sizing+"f.rudy"
ising_o_name = "test"+sizing+"o.rudy"
mps_name     = "test"+sizing+"o.mps"

prob.exportIsing(ising_f_name, feasibility=True)
prob.exportIsing(ising_o_name, feasibility=False)
prob.exportMPS(mps_name)


#%%

#prob.exportQUBO('ExMIRPg2.qubo')
#prob.exportIsing('ExMIRPg2.rudy')


#%%
# Test sampler in RoutingProblem

#sampler = rp.getSampledKey
#testD = { 'a':100, 'b':101, 'c':102, 'd':10 }
#counts = { k:0 for k in testD.keys() }
#N = 10000
#for i in range(N):
#    k,foo = sampler(testD,extent=1)
#    counts[k] = counts[k] + 1
#print(counts)