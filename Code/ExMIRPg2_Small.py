# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:38:19 2018

@author: smharwo
"""
"""
Adapting something that looks like a MIRPLib instance to fit the VRPTW formalism
Specifically, looking at LR1_DR02_VC01_V6a from
https://mirplib.scl.gatech.edu/sites/default/files/LR1_DR02_VC01_V6a.txt
This is one of the first "group 2" instances

DEPRECATED- see ExMIRPg2; has adjustable time horizon
Six identical vessels, one supply port, two demand ports
The main challenge with this instance is figuring out the time windows that
sort of respect the inventory bounds and delivery sizes.
For now, that has been taken care of "offline"
"""

import numpy
import RoutingProblem as rp

prob = rp.RoutingProblem()

# Set vehicle capacity to ZERO
# The trick we will employ is to give supply regions "negative" demand,
# so that the overall running total of demand met for a valid route will always be nonpositive
prob.setVehicleCap(0)
CargoSize = 300

# Add Nodes
# 1 Supply, 2 Demand ports
# but multiple nodes for each, since expanded in time

# Supply:
Supply = [None]*8
for s in range(len(Supply)):
    TW = (2+4*s, 3+4*s)
    Supply[s] = 'S {},{}'.format(TW[0],TW[1])
    prob.addNode(Supply[s],-CargoSize,TW)

# Demand 1
Demand1 = [None]*4
for d1 in range(len(Demand1)):
    TW = (4+9*d1,6+9*d1)
    Demand1[d1] = 'D1 {},{}'.format(TW[0],TW[1])
    prob.addNode(Demand1[d1],CargoSize,TW)

# Demand 2
Demand2 = [None]*4
for d2 in range(len(Demand2)):
    TW = (2+8*d2,5+8*d2)
    Demand2[d2] = 'D2 {},{}'.format(TW[0],TW[1])
    prob.addNode(Demand2[d2],CargoSize,TW)

# Dummy nodes:
# A depot is required, but we use it and some dummy Supply nodes to control
# where vessels are initially available
prob.addNode('Depot',0)
prob.setDepot('Depot')
prob.addNode('Dum1',-CargoSize)
prob.addNode('Dum2',-CargoSize)
prob.addNode('Dum3',-CargoSize)

print(prob.NameList)



# Arcs
# Regular supply/demand nodes are fully connected,
# have simple travel times based on the location
# (and including any loading/unloading times)
# Costs include port fees, etc, but note they are not symmetric
# (because the nodes have a time component, not all arcs are physically reasonable-
# but checking for a feasible/valid route will catch that)
for s in Supply:
    for d in Demand1:
        prob.addArc(s,d, 8, 475)
        prob.addArc(d,s, 8, 511)
for s in Supply:
    for d in Demand2:
        prob.addArc(s,d, 14, 880)
        prob.addArc(d,s, 14, 908)
## We are not allowing split deliveries, so
## travel directly between demand ports will never happen
## but we include anyway
#for d1 in Demand1:
#    for d2 in Demand2:
#        prob.addArc(d1,d2, 17, 1011)
#        prob.addArc(d2,d1, 17, 1003)

# Special arcs to enforce initial conditions:
# Vessels starting at supply ports
# vessel 4:
prob.addArc('Depot',Supply[0],2,0)
# vessel 1:
prob.addArc('Depot',Supply[1],5,0)
# vessel 3:
prob.addArc('Depot',Supply[2],8,0)

# Vessels starting at demand ports, but which have initial loadings
# must go through a dummy port
# vessel 0:
prob.addArc('Depot','Dum1',0)
prob.addArc('Dum1',Demand2[0],0,0)
# vessel 2:
prob.addArc('Depot','Dum2',0)
prob.addArc('Dum2',Demand2[1],10,0)
# vessel 5:
prob.addArc('Depot','Dum3',0)
prob.addArc('Dum3',Demand1[0],4,0)


# Exiting arcs (back to Depot)
# no time, but at a high cost, to discourage premature exit
# For simplicity, allow exit from any "regular" supply/demand node
for s in Supply:
    prob.addArc(s,'Depot', 0, 2000)
for d in Demand1:
    prob.addArc(d,'Depot', 0, 2000)
for d in Demand2:
    prob.addArc(d,'Depot', 0, 2000)

#%%

## test:
#r = ['Depot',Supply[0],Demand1[1],Supply[5],Demand1[3],'Depot']
#f,c,vn = prob.checkRoute(r)
#print(f)
#print(vn)

#r = prob.getRouteNames(prob.generateRoutes())
#print(r)
#print(prob.checkRoute(r))

# Add routes
numpy.random.seed(0) # for reproducibility
NodesVisited = [0]*len(prob.NameList)
vf = [0]*len(prob.NameList)
for i in range(1000):
    r,vf = prob.generateRoute(vf)
    f,v = prob.addRoute(r)
    NodesVisited = [ a or b for a,b in zip(v,NodesVisited) ]

if all(NodesVisited):
    print('All nodes covered!')
else:
    print('Some node not covered by a route')

#%%

prob.exportQUBO('ExMIRPg2_Small.qubo')

#%%

prob.exportIsing('ExMIRPg2_Small.rudy')


#%%
## tests sampler
#sampler = rp.getSampledKey
#testD = { 'a':100, 'b':101, 'c':102, 'd':103 }
#counts = { k:0 for k in testD.keys() }
#N = 10000
#for i in range(N):
#    k = sampler(testD)
#    counts[k] = counts[k] + 1
#print(counts)