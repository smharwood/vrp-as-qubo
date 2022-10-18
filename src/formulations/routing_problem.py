"""
SM Harwood
16 October 2022
"""
import numpy as np
from vrptw import VRPTW

class RoutingProblem:
    """
    A base class intended to enforce a consistent interface among the different
    formulations of the Vehicle Routing Problem with Time Windows (VRPTW)
    """

    def __init__(self, vrptw):
        """
        Constructor

        Parameters:
        vrptw (VRPTW): an optional VRPTW object to hold the underlying graph data
            of the problem. If None, a new object is created.
        """
        if vrptw is None:
            vrptw = VRPTW()
        self.vrptw = vrptw
        self.feasible_solution = None

    @property
    def nodes(self):
        """ The nodes of the underlying VRPTW problem """
        return self.vrptw.nodes

    @property
    def node_names(self):
        """ The names of the nodes of the underlying VRPTW problem """
        return self.vrptw.node_names

    @property
    def arcs(self):
        """ The arcs of the underlying VRPTW problem """
        return self.vrptw.arcs

    @property
    def depot_index(self):
        """ The index of the depot node """
        return self.vrptw.depot_index

    @property
    def vehicle_cap(self):
        """ The capacity of the vehicles """
        return self.vrptw.vehicle_cap

    @property
    def initial_loading(self):
        """ The initial loading of vehicles when leaving depot """
        return self.vrptw.initial_loading

    def set_vehicle_cap(self, vehicle_cap):
        return self.vrptw.set_vehicle_cap(vehicle_cap)

    def set_initial_loading(self, loading):
        return self.vrptw.set_initial_loading(loading)

    def add_node(self, node_name, demand, t_w=(0, np.inf)):
        return self.vrptw.add_node(node_name, demand, t_w)

    def get_node_index(self, node_name):
        return self.vrptw.get_node_index(node_name)

    def get_node(self, node_name):
        return self.vrptw.get_node(node_name)

    def set_depot(self, depot_name):
        """ Set node with name `depot_name` as depot node """
        return self.vrptw.set_depot(depot_name)

    def add_arc(self, origin_name, destination_name, travel_time, cost=0):
        """
        Add a potentially allowable arc

        Return:
            added (bool): whether arc was added or not
        """
        return self.vrptw.add_arc(origin_name, destination_name, travel_time, cost)

    def estimate_max_vehicles(self):
        """ Estimate maximum number of vehicles """
        return self.vrptw.estimate_max_vehicles()

    def get_num_variables(self):
        """ Get the number of variables in this formulation """
        raise NotImplementedError

    def make_feasible(self, high_cost):
        """
        Some sort of greedy construction heuristic to make sure the problem is
        feasible. We add dummy node/arcs as necessary to emulate more
        vehicles being available.
        """
        raise NotImplementedError

    def get_constraint_data(self):
        """
        Return constraints in a consistent way
        A_eq * x = b_eq
        xᵀ * Q_eq * x = r_eq

        Parameters:

        Return:
            A_eq (array): 2-d array of linear equality constraints
            b_eq (array): 1-d array of right-hand side of equality constraints
            Q_eq (array): 2-d array of a single quadratic equality constraint
                (potentially all zeros if there are no nontrivial quadratic constraints)
            r_eq (float): Right-hand side of the quadratic constraint
        """
        raise NotImplementedError

    def get_qubo(self, penalty_parameter=None, feasibility=False):
        """
        Get the Quadratic Unconstrained Binary Optimization problem reformulation

        args:
        penalty_parameter (float): value of penalty parameter to use for reformulation.
            If None, it is determined automatically
        feasibility (bool): Get the feasibility problem (ignore the objective)

        Return:
        Q (ndarray): Square matrix defining QUBO
        c (float): a constant that makes the objective of the QUBO equal to the
            objective value of the original constrained integer program
        """
        raise NotImplementedError

    def get_cplex_prob(self):
        """
        Get a CPLEX object containing the original constrained integer program
        representation

        args:
        None

        Return:
        cplex_prob (cplex.Cplex): A CPLEX object defining the MIP problem
        """
        raise NotImplementedError

    def export_mip(self, filename=None):
        """ Export constrained integer program representation of problem """
        if filename is None:
            filename = f"{type(self).__name__}.lp"
        cplex_prob = self.get_cplex_prob()
        cplex_prob.write(filename)
        return

    def solve_cplex_prob(self, filename_sol="cplex.sol"):
        """ Solve constrained integer formulation with CPLEX """
        cplex_prob = self.get_cplex_prob()
        cplex_prob.solve()
        cplex_prob.solution.write(filename_sol)
        return
