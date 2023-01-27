"""
SM Harwood
19 October 2022

Simple example from
Desrochers, Desrosiers, Solomon, "A new optimization algorithm for the vehicle
routing problem with time windows"
to test stuff
"""
import numpy as np
from ..routing_problem import (
    VRPTW,
    ArcBasedRoutingProblem,
    PathBasedRoutingProblem,
    SequenceBasedRoutingProblem
)

def get_vrptw():
    """ Get the basic VRPTW data """
    # Get object
    vrp = VRPTW()
    vrp.set_vehicle_cap(6)
    vrp.set_initial_loading(6)

    # A depot node is required
    vrp.add_node('D', 0, (0,np.inf))
    vrp.set_depot('D')

    # Add nodes (Name, Demand, Time Window)
    vrp.add_node('1', 1, (1,7))
    vrp.add_node('2', 2, (2,4))
    vrp.add_node('3', 2, (4,7))

    # Add arcs (Origin, Destination, Time, Cost)
    vrp.add_arc('D','1', 1, 1)
    vrp.add_arc('D','2', 2, 2)
    vrp.add_arc('D','3', 2, 2)

    vrp.add_arc('1','D', 1, 1)
    vrp.add_arc('1','2', 1, 1)
    vrp.add_arc('1','3', 1, 1)

    vrp.add_arc('2','D', 2, 2)
    vrp.add_arc('2','1', 1, 1)
    vrp.add_arc('2','3', 1, 1)

    vrp.add_arc('3','D', 2, 2)
    vrp.add_arc('3','1', 1, 1)

    return vrp

def get_high_cost():
    """ Get a "high cost" route for this problem
    (any feasible route has cost <= 8)
    """
    return 10

def get_arc_based(time_points=None):
    """ Get the arc-based formulation """
    if time_points is None:
        time_points = [0,1,2,4,7]

    vrp = get_vrptw()
    abrp = ArcBasedRoutingProblem(vrp)
    abrp.add_time_points(time_points)
    return abrp

def get_path_based():
    """ Get the path-based formulation """
    vrp = get_vrptw()
    pbrp = PathBasedRoutingProblem(vrp)

    # Add/check routes
    # From paper, we know there are 11
    routes = [None]*13
    routes[0] = ['D','1','D']
    routes[1] = ['D','2','D']
    routes[2] = ['D','3','D']
    routes[3] = ['D','1','2','D']
    routes[4] = ['D','1','3','D']
    routes[5] = ['D','2','1','D']
    routes[6] = ['D','2','3','D']
    routes[7] = ['D','3','1','D']
    routes[8] = ['D','1','2','3','D']
    routes[9] = ['D','2','1','3','D']
    routes[10]= ['D','2','3','1','D']
    routes[11]= ['D','2','3','1','3','D'] # infeasible to test
    routes[12]= ['D'] # is this feasible? no

    for route in routes:
        f, _ = pbrp.add_route(route)
        if not f:
            print(f"{route} not feasible")
    return pbrp

def get_sequence_based(max_vehicles=2, max_sequence_length=4):
    """ Get the sequence-based formulation """
    vrp = get_vrptw()
    sbrp = SequenceBasedRoutingProblem(vrp)
    sbrp.set_max_vehicles(max_vehicles)
    sbrp.set_max_sequence_length(max_sequence_length)
    return sbrp
