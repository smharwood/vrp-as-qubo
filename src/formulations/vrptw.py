"""
SM Harwood
17 October 2022
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VRPTW:
    """
    A class to organize the basic data of a
    Vehicle Routing Problem with Time Windows (VRPTW)
    This includes the graph structure of the problem and vehicle sizing.

    This formulation of the problem is based on
    M. Desrochers, J. Desrosiers, and M. Solomon, "A new optimization algorithm
    for the vehicle routing problem with time windows"
    https://doi.org/10.1287/opre.40.2.342
    """
    def __init__(self):
        self.node_names = []
        self.nodes = []
        self.arcs = dict()
        self.depot_index = 0 # default depot is zeroth node
        self.vehicle_cap = None
        self.initial_loading = None

    def set_vehicle_cap(self, vehicle_cap):
        """ Set the capacity of the vehicles """
        self.vehicle_cap = vehicle_cap
        return

    def set_initial_loading(self, loading):
        """ Set the load size of vehicles when they leave the depot """
        self.initial_loading = loading
        return

    def add_node(self, node_name, demand, t_w=(0, np.inf)):
        """
        Add a node to the problem,
        with demand level `demand` and time window `t_w`
        """
        assert node_name not in self.node_names, node_name + " is already in Node List"
        self.nodes.append(Node(node_name, demand, t_w))
        self.node_names.append(node_name)
        return

    def get_node_index(self, node_name):
        """ Get the index of the node `node_name` """
        return self.node_names.index(node_name)

    def get_node(self, node_name):
        """ Get the node `node_name` """
        return self.nodes[self.get_node_index(node_name)]

    def set_depot(self, depot_name):
        """ Take node named `depot_name` and move to first position in list """
        d_index = self.node_names.index(depot_name)
        if not np.isinf(self.nodes[d_index].time_window[1]):
            logger.warning("Consider making Depot time window infinite in size...")
        if d_index == 0:
            return
        depot = self.nodes.pop(d_index)
        self.node_names.remove(depot_name)
        self.nodes.insert(0, depot)
        self.node_names.insert(0, depot_name)
        return

    def add_arc(self, origin_name, destination_name, travel_time, cost=0):
        """
        Add a potentially allowable arc;
        we also check feasibility of TIMING:
            allow if origin time window start plus travel time <= destination time window end
            (There is no way this arc could ever be used if this is not satisfied)

        Return:
            added (bool): whether arc was added or not
        """
        i = self.get_node_index(origin_name)
        j = self.get_node_index(destination_name)
        # self.arcs[(i, j)] = Arc(self.nodes[i], self.nodes[j], travel_time, cost)
        # Add arc if timing works:
        departure_time = self.nodes[i].time_window[0]
        if departure_time + travel_time <= self.nodes[j].time_window[1]:
            self.arcs[(i, j)] = Arc(self.nodes[i], self.nodes[j], travel_time, cost)
            return True
        #else
        return False

    def estimate_max_vehicles(self):
        """
        Estimate maximum number of vehicles based on degree of depot node in
        the graph. Since one basic constraint of the VRPTW is that each node is
        visited exactly once, and all vehicles start at the depot, the number of
        vehicles is limited by this degree.
        """
        num_outgoing = 0
        num_incoming = 0
        for arc in self.arcs:
            if arc[0] == self.depot_index:
                num_outgoing += 1
            if arc[1] == self.depot_index:
                num_incoming += 1
        return min(num_outgoing, num_incoming)


class Node:
    """
    A node is a customer, with a demand level that must be satisfied in a particular
    window of time.
    Here, the sign convention for demand is from the perspective of the node:
        positive demand must be delivered,
        negative demand is supply that must be picked up.
    """
    def __init__(self, name, demand, t_w):
        assert t_w[0] <= t_w[1], f"Time window for {name} not valid: {t_w[0]} > {t_w[1]}"
        self.name = name
        self.demand = demand
        self.time_window = t_w

    def get_name(self):
        """ Get the name of this node """
        return self.name

    def get_demand(self):
        """ Return the demand level of the node """
        return self.demand

    def get_load(self):
        """
        Return what is loaded onto/off a vehicle servicing this node
        (i.e., load is negative demand)
        """
        return -self.demand

    def get_window(self):
        """ Return the time window (a, b) of the node """
        return self.time_window

    def __str__(self):
        return f"{self.name}: {self.demand} in {self.time_window}"


class Arc:
    """
    An arc goes from one node to another (distinct) node
    It has an associated travel time, and potentially a cost
    """
    def __init__(self, origin, destination, travel_time, cost):
        """Define Arc by origin and destination Nodes and time/cost"""
        assert origin is not destination, "Arc endpoints must be distinct"
        self.origin = origin
        self.destination = destination
        self.travel_time = travel_time
        self.cost = cost

    def get_origin(self):
        """ Get the origin node of this arc """
        return self.origin

    def get_destination(self):
        """ Get the destination node of this arc """
        return self.destination

    def get_travel_time(self):
        """ Get the travel time of this arc """
        return self.travel_time

    def get_cost(self):
        """ Get the cost of this arc """
        return self.cost

    def __str__(self):
        return f"{self.origin.name} to {self.destination.name}, t={self.travel_time:.2f}"
