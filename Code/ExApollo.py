# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 08:18:01 2018

@author: smharwo
"""
"""
APOLLO as RoutingProblem, 
JUST TO GET A SENSE OF SIZE
Not meant to be fully functional (need arcs)
"""

#%%
import RoutingProblem as rp

# Create a routing problem
prob = rp.RoutingProblem()

# Set vehicle capacity to ZERO
# The trick we will employ is to give supply regions "negative" demand,
# so that the overall running total of demand met for a valid route will always be nonpositive
prob.setVehicleCap(0)
CargoSize = 200
TimeHorizon = 365

# A depot node is required (zero demand)
prob.addNode('Depot',0)
prob.setDepot('Depot')

#%%
# Add Supply Nodes

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
    
SupplyLists = []
Name = 'Gorgon'
tankage = 348+287
initial = 150+25
rate = 9.4+5
SupplyLists.append(addSupplyNodes(prob,Name,initial,rate,CargoSize,tankage))

Name = 'PNGLNG'
tankage = 500+500
initial = 300+400
rate = 2*20.5
SupplyLists.append(addSupplyNodes(prob,Name,initial,rate,CargoSize,tankage))

#%%
# Add Demand Nodes

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
    
DemandLists = []
Name = 'PNGLNG-CPC'
# Yung-An
tankage = 300
initial = 220
rate = 7.55
DemandLists.append(addDemandNodes(prob,Name,initial,-rate,CargoSize,tankage))

Name = 'PNGLNG-TEPCO'
# Futtsu
tankage = 300
initial = 200
rate = 3.74
DemandLists.append(addDemandNodes(prob,Name,initial,-rate,CargoSize,tankage))

Name = 'PNGLNG-OsakaGas'
# Himeji
tankage = 250
initial = 150
rate = 3.13
DemandLists.append(addDemandNodes(prob,Name,initial,-rate,CargoSize,tankage))

Name = 'PNGLNG-SinopecGroup'
# Shandong
tankage = 500
initial = 300
rate = 12.58
DemandLists.append(addDemandNodes(prob,Name,initial,-rate,CargoSize,tankage))

Name = 'EM-PetroChina'
# Fuqing
tankage = 222
initial = 150
rate = 7.42
DemandLists.append(addDemandNodes(prob,Name,initial,-rate,CargoSize,tankage))

#%%
# Would define and add Arcs here

# Special arcs to enforce initial conditions

# Exiting arcs (back to Depot)

#%%
print("Number of nodes: {}".format(len(prob.Nodes)))
# Number of possible routes can be figured out with a back-of-the-envelope calc...