# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:37:45 2019

@author: smharwo

Special Test instances for Stanford/Cornell:
vary TimeHorizon and number of routes added to get different size problems
"""

import numpy as np
import ExMIRPg2 as ex

#%%
# TIME HORIZON = 40; expect 21 nodes, 56 variables
TH = 40
prob = ex.DefineProblem(TH)

print('Number of nodes/constraints: {}'.format(len(prob.Nodes)-1))
for n in prob.Nodes:
    print(n)

# Add routes
# This will automatically generate up to the given number of routes
np.random.seed(0) # for reproducibility
# Play with route generation; try one or the other or both
vf = prob.addRoutes(50,explore=10)
vf = prob.addRoutes(50,explore=1,vf=vf)
vf = prob.addRoutes(50,explore=0,vf=vf)
prob.addRoutesFeasibility(50,explore=10) 
prob.addRoutesFeasibility(50,explore=1) 
prob.addRoutesFeasibility(50,explore=0) 

print('Number of variables/routes: {}'.format(len(prob.mip_variables)))

#%%
sizing  = "_{}_{}_".format(len(prob.Nodes)-1, len(prob.mip_variables))
ising_f_name = "test"+sizing+"f.rudy"
ising_o_name = "test"+sizing+"o.rudy"
mps_name     = "test"+sizing+"o.mps"

prob.exportIsing(ising_f_name, feasibility=True)
prob.exportIsing(ising_o_name, feasibility=False)
prob.exportMPS(mps_name)


#%%
# TIME HORIZON = 50; expect 24 nodes, 100 variables
TH = 50
prob = ex.DefineProblem(TH)

print('Number of nodes/constraints: {}'.format(len(prob.Nodes)-1))
for n in prob.Nodes:
    print(n)

#%%
# Add routes
# This will automatically generate up to the given number of routes
np.random.seed(0) # for reproducibility
# Play with route generation; try one or the other or both
vf = prob.addRoutes(50,explore=10)
vf = prob.addRoutes(50,explore=1,vf=vf)
vf = prob.addRoutes(50,explore=0,vf=vf)
prob.addRoutesFeasibility(50,explore=10) 
prob.addRoutesFeasibility(50,explore=1) 
prob.addRoutesFeasibility(50,explore=0) 

print('Number of variables/routes: {}'.format(len(prob.mip_variables)))

# Kludge: remove nodes/constraints to get feasible problem
prob.ignoreNode('D1-3')
prob.ignoreNode('S-7')

#%%
sizing  = "_{}_{}_".format(len(prob.Nodes)-1, len(prob.mip_variables))
ising_f_name = "test"+sizing+"f.rudy"
ising_o_name = "test"+sizing+"o.rudy"
mps_name     = "test"+sizing+"o.mps"

prob.exportIsing(ising_f_name, feasibility=True)
prob.exportIsing(ising_o_name, feasibility=False)
prob.exportMPS(mps_name)