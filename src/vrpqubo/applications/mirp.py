"""
SM Harwood
17 October 2022
"""
from itertools import product
import numpy as np
from ..formulations import (
    VRPTW,
    ArcBasedRoutingProblem,
    PathBasedRoutingProblem,
    SequenceBasedRoutingProblem
)

class MIRP:
    """
    A class to help implement a Maritime Inventory Routing Problem
    cast as a Vehicle Routing Problem with Time Windows
    """
    def __init__(
            self,
            cargo_size,
            time_horizon
        ):
        self.cargo_size = cargo_size
        self.time_horizon = time_horizon
        self.supply_ports = []
        self.demand_ports = []
        self.port_mapping = dict()
        self.port_frequency = dict()
        self.routes_added = False
        self.vrptw = VRPTW()
        # The way we fit a MIRP into the VRPTW form is to assume that the depot
        # is essentially a dummy/"source" node, and vessels have zero initial loading.
        # Also, recall that the VRPTW problem assumes that vessels/vehicles are
        # homogeneous (same size).
        # We also assume full load/unload of vessels at each node.
        self.vrptw.set_initial_loading(0)
        self.vrptw.set_vehicle_cap(cargo_size)
        self.vrptw.add_node("Depot", 0)
        self.vrptw.set_depot("Depot")

        # The various formulation objects to be built later
        self.abrp = None
        self.pbrp = None
        self.sbrp = None

    def add_node(self, name, demand, time_window):
        """ Add a node to the underlying VRPTW graph """
        return self.vrptw.add_node(name, demand, time_window)

    def add_arc(self, origin, destination, travel_time, cost):
        """ Add an arc to the underlying VRPTW graph """
        return self.vrptw.add_arc(origin, destination, travel_time, cost)

    def get_time_window(self,
            num_prior_visits,
            inventory_init,
            inventory_rate,
            inventory_cap,
        ):
        """
        Determine the time window of this node. A single physical port must be
        visited multiple times as it runs out of inventory or fills up its storage
        capacity. The time window in which it must be visited depends on the number
        of times it has previously been visited

        Parameters:
        num_prior_visits (int): Number of times this node has been visited/serviced
            aready
        inventory_init (float): Initial inventory level
        inventory_rate (float): Rate of change of inventory.
            inventory_rate > 0: This is a supply port
            inventory_rate < 0: This is a demand port
        inventory_cap (float): Amount of inventory capacity at this port

        return:
        tw_start (float): Start of time window
        tw_end (float): End of time window
        """
        size = self.cargo_size
        # inventory(t) = inventory_init + t*inventory_rate
        if inventory_rate > 0:
            # SUPPLY
            # Earliest a ship can load a full shipload:
            # inventory(t) - (num_prior_visits+1)*size >= 0
            tw0 = ((num_prior_visits+1)*size - inventory_init)/inventory_rate
            # latest a ship can arrive before port capacity is exceeded:
            # inventory(t) - (num_prior_visits)*size > inventory_cap
            tw1 = (inventory_cap + (num_prior_visits)*size - inventory_init)/inventory_rate
            return (tw0, tw1)
        else:
            # DEMAND
            # Earliest a ship can discharge a full load into inventory:
            # inventory(t) + (num_prior_visits+1)*size <= inventory_cap
            tw0 = (inventory_cap - (num_prior_visits+1)*size - inventory_init)/inventory_rate
            # latest a ship can arrive before port runs out of inventory:
            # inventory(t) + (num_prior_visits)*size < 0
            tw1 = (-(num_prior_visits)*size - inventory_init)/inventory_rate
            return (tw0, tw1)

    def add_nodes(self,
            name,
            inventory_init,
            inventory_rate,
            inventory_cap
        ):
        """
        Add nodes for this supply or demand port. A single physical port must be
        visited multiple times as it runs out of inventory or fills up its storage
        capacity

        Parameters:
        name (string): Base name of this port
        inventory_init (float): Initial inventory level
        inventory_rate (float): Rate of change of inventory.
            inventory_rate > 0: This is a supply port
            inventory_rate < 0: This is a demand port
        inventory_cap (float): Amount of inventory capacity at this port

        Return:
        node_names (list of string): The names of the nodes that were added
        """
        if inventory_rate > 0:
            # Supply port. "Demand" is negative, equal to full ship loading
            demand_level = -self.cargo_size
            self.supply_ports.append(name)
        else:
            # Demand port. Demand is positive, equal to full ship unloading
            demand_level = self.cargo_size
            self.demand_ports.append(name)

        node_names = []
        self.port_mapping[name] = []
        # Port frequency is an estimate of how frequently this port must be visited
        self.port_frequency[name] = np.fabs(inventory_cap/inventory_rate)
        num_prior_visits = 0
        while True:
            t_w = self.get_time_window(
                num_prior_visits,
                inventory_init,
                inventory_rate,
                inventory_cap
            )
            # only time windows fully within time horizon are added
            if t_w[1] > self.time_horizon:
                break
            # otherwise the time window is within the time horizon
            node_names.append(f"{name}-{num_prior_visits}")
            self.add_node(node_names[-1], demand_level, t_w)
            self.port_mapping[name].append(node_names[-1])
            num_prior_visits+=1
        return node_names

    def add_travel_arcs(self,
            distance_function,
            vessel_speed,
            cost_per_unit_distance,
            supply_port_fees,
            demand_port_fees
        ):
        """
        Add main travel arcs between any supply port and any demand port (and vice versa)
        Time and cost based on distance, costs include port fees.
        Because the nodes have a time component, not all arcs are physically reasonable
         - but the underlying VRPTW checks for that.

        Parameters:
        """
        for s_p in self.supply_ports:
            for d_p in self.demand_ports:
                distance = distance_function(s_p, d_p)
                travel_time = distance/vessel_speed
                travel_cost = distance*cost_per_unit_distance
                for s_node, d_node in product(self.port_mapping[s_p], self.port_mapping[d_p]):
                    self.add_arc(s_node, d_node, travel_time, travel_cost + demand_port_fees[d_p])
                    self.add_arc(d_node, s_node, travel_time, travel_cost + supply_port_fees[s_p])
        return

    def add_entry_arcs(self, time_limit, travel_time=0, cost=0):
        """
        Add entry arcs from depot to any Supply node with time window less than
        `time_limit`

        Early demand ports will not get visited in time;
        Need to assume that there are loaded vessels available at start of time horizon.
        Enforce with dummy supply nodes.
        Note this is NOT required by all formulations
        """
        depot_name = self.vrptw.node_names[self.vrptw.depot_index]
        for port in self.supply_ports:
            for node in self.port_mapping[port]:
                if self.vrptw.get_node(node).time_window[1] < time_limit:
                    self.add_arc(depot_name, node, travel_time, cost)

        num_dum = 0
        for port in self.demand_ports:
            for node in self.port_mapping[port]:
                if self.vrptw.get_node(node).time_window[1] < time_limit:
                    dummy = f"Dum{num_dum}"
                    num_dum += 1
                    self.vrptw.add_node(dummy, -self.cargo_size)
                    self.add_arc(depot_name, dummy, 0, 0)
                    self.add_arc(dummy, node, travel_time, cost)
        return

    def add_exit_arcs(self, travel_time=0, cost=0):
        """
        Add exit arcs (back to Depot) from any "regular" supply/demand node
        """
        depot_name = self.vrptw.node_names[self.vrptw.depot_index]
        for port in self.supply_ports + self.demand_ports:
            for node in self.port_mapping[port]:
                self.add_arc(node, depot_name, travel_time, cost)
        return

    def estimate_high_cost(self):
        """
        Estimate a "high cost" as approximately the cost of a very expensive route.
        Look at the port that must be visited the most often, then multiply the
        number of times it must be visited by twice the most expensive arc cost
        """
        most_freq = min(self.port_frequency.values())
        most_trips = self.time_horizon/most_freq
        costs = [arc.get_cost() for arc in self.vrptw.arcs.values()]
        return 2*max(costs)*most_trips

    def get_arc_based(self, make_feasible=True):
        """
        Return the arc-based routing problem object
        Build if necessary, including defining the time periods in some way
        """
        if self.abrp is not None:
            return self.abrp
        self.abrp = ArcBasedRoutingProblem(self.vrptw)
        # The only other thing we need to do for arc-based formulation is add
        # time points. This can be tricky; we want as much resolution as possible,
        # but also keep it small
        # Options:
        #   integers that fall in any node's time window
        #   endpoints and midpoints of time windows
        # We will use the former
        tw_points = []
        for n in self.vrptw.nodes:
            (tw0, tw1) = n.get_window()
            if np.isinf(tw1):
                continue
            tw_points += np.arange(np.ceil(tw0), np.floor(tw1)+1).tolist()
        tw_points.append(0)
        tw_points = set(tw_points) # get unique values
        timepoints = list(tw_points)
        timepoints.sort()
        self.abrp.add_time_points(timepoints)

        if make_feasible:
            high_cost = self.estimate_high_cost()
            self.abrp.make_feasible(high_cost)
        return self.abrp

    def get_path_based(self, make_feasible=True):
        """
        Return the path-based routing problem object
        Build if necessary, including constructing routes
        """
        if self.pbrp is not None:
            return self.pbrp
        self.pbrp = PathBasedRoutingProblem(self.vrptw)

        # for reproducibility:
        np.random.seed(0)
        # Make the depot high cost, to discourage premature exit,
        # make later arrival times more expensive, to favor early arrival
        high_cost = self.estimate_high_cost()
        def time_costs(t):
            if t <=10:
                return 0
            #else
            return 100*t
        node_costs = [0]*len(self.vrptw.nodes)
        node_costs[self.vrptw.depot_index] = high_cost

        for (explore, rep) in zip([0.0, 1.0, np.inf],
                                  [1, int(self.time_horizon), int(10*self.time_horizon)]):
            for _ in range(rep):
                self.pbrp.add_routes_better(explore, node_costs, time_costs)

        if make_feasible:
            self.pbrp.make_feasible(high_cost)
        return self.pbrp

    def get_sequence_based(self, make_feasible=True, strict=True):
        """
        Return the sequence-based routing problem object
        Build if necessary, including setting number of vehicles and sequence numbers
        """
        if self.sbrp is not None:
            return self.sbrp
        self.sbrp = SequenceBasedRoutingProblem(self.vrptw, strict)
        max_vehicles = self.vrptw.estimate_max_vehicles()
        self.sbrp.set_max_vehicles(max_vehicles)
        # Number of moves/stops in a route:
        # estimate from time horizon divided by shortest travel arc, plus entry and exit
        travel_times = [arc.get_travel_time() for arc in self.vrptw.arcs.values()]
        min_travel_time = min(filter(lambda t: t > 0, travel_times))
        self.sbrp.set_max_sequence_length(int(self.time_horizon/min_travel_time + 2))
        if make_feasible:
            high_cost = self.estimate_high_cost()
            self.sbrp.make_feasible(high_cost)
        return self.sbrp
