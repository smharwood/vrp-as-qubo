"""
SM Harwood
22 June 2018

Simple example from
Desrochers, Desrosiers, Solomon, "A new optimization algorithm for the vehicle routing problem with time windows"
to test stuff
"""
import path_based.RoutingProblem as rp

def DefineProblem():
    prob = rp.RoutingProblem()

    # Set vehicle capacity + initial loading of vehicle when it leaves depot
    prob.setVehicleCap(6)
    prob.setInitialLoading(6)

    # Add nodes (Name, Demand, Time Window)
    prob.addNode('D',0)
    prob.addNode('1',1,(1,7))
    prob.addNode('2',2,(2,4))
    prob.addNode('3',2,(4,7))
    prob.setDepot('D')

    # Add arcs (Origin, Destination, Time, Cost=0)
    prob.addArc('D','1',1)
    prob.addArc('D','2',2)
    prob.addArc('D','3',2)

    prob.addArc('1','D',1)
    prob.addArc('1','2',1)#,100)
    prob.addArc('1','3',1)#,100)

    prob.addArc('2','D',2)
    prob.addArc('2','1',1)#,100)
    prob.addArc('2','3',1)#,100)

    prob.addArc('3','D',2)
    prob.addArc('3','1',1)#,100)


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
    R[11]= ['D','2','3','1','3','D'] # infeasible to test
    R[12]= ['D'] # is this feasible? no

    for route in R:
        f, _ = prob.addRoute(route)
        if not f:
            print(str(route)+' not feasible')

    return prob

def test_sampler():
    """ Test sampler in RoutingProblem """
    sampler = rp.getSampledKey
    testD = { 'a':100, 'b':101, 'c':102, 'd':10 }
    counts = { k:0 for k in testD.keys() }
    N = 10000
    for i in range(N):
        k, _ = sampler(testD,extent=1)
        counts[k] = counts[k] + 1
    print(counts)