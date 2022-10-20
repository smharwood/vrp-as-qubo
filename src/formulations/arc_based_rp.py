"""
SM Harwood
19 December 2019
"""
import time
import logging
import numpy as np
from scipy import sparse
from .routing_problem import RoutingProblem
try:
    import cplex
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ArcBasedRoutingProblem(RoutingProblem):
    """
    Class to implement the arc-based formulation of a VRPTW
    Base math program is a 0-1 Integer Linear Program
    which can be transformed to a
    Quadratic Unconstrained Binary Optimization (QUBO) problem
    """

    def __init__(self, vrptw=None):
        super().__init__(vrptw)
        self.time_points = []
        self.var_mapping = []
        self.num_variables = 0
        self.variables_enumerated = False
        self.constraints_built = False
        self.objective_built = False
        self.constraint_names = []

        # Parameters of the constrained math program formulation:
        self.objective = None
        self.constraints_matrix = None
        self.constraints_rhs = None

    def add_time_points(self, time_points):
        """ Populate the valid time points """
        # Copy the SORTED timepoints; this can be a sequence, a np array, whatever.
        # Variables are indexed by a sequence which can be made of anything,
        # but we do assume they are sorted for some convenience
        self.time_points = np.sort(time_points)
        return

    def check_arc(self, arc_key):
        """ Is the arc valid? """
        return arc_key in self.arcs
        # try:
        #     self.arcs[arc_key]
        #     return True
        # except KeyError:
        #     return False

    def check_node_time_compat(self, node_index, time_point):
        """ is this node-time pair compatible? """
        t_w = self.nodes[node_index].get_window()
        return t_w[0] <= time_point <= t_w[1]

    def enumerate_variables(self):
        """ Enumerate/map the variables """
        if self.variables_enumerated:
            return
        start = time.time()
        self.enumerate_variables_quicker()
        #self.enumerate_variables_exhaustive()
        duration = time.time() - start
        logger.info("Variable enumeration took %s seconds", duration)
        return

    def enumerate_variables_quicker(self):
        """ Basic operation that needs to be done to keep track of variable counts, indexing """
        num_vars = 0
        # Loop over (i,s,j,t)
        # and check if a variable is allowed (nonzero)
        # Simplification: loop over arcs
        for (i,j) in self.arcs.keys():
            for s in self.time_points:
                # First check:
                # is s in the time window of i?
                if s < self.nodes[i].get_window()[0]:
                    continue
                if s > self.nodes[i].get_window()[1]:
                    # time_points is sorted, so s will keep increasing in this loop
                    # This condition will continue to be satisfied
                    break
                for t in self.time_points:
                    # Second check:
                    # is t in the time window of j?
                    if t < self.nodes[j].get_window()[0]:
                        continue
                    if t > self.nodes[j].get_window()[1]:
                        # Same as with s loop: t will keep increasing in this loop
                        break
                    # Third check
                    # is the travel time from i to j consistent with the timing?
                    if s + self.arcs[(i,j)].get_travel_time() > t:
                        continue

                    # at this point, the tuple (i,s,j,t) is allowed
                    # record the index
                    self.var_mapping.append((i,s,j,t))
                    num_vars += 1
        # end loops
        self.num_variables = num_vars
        self.variables_enumerated = True
        return

    def enumerate_variables_exhaustive(self):
        """ Basic operation that needs to be done to keep track of variable counts, indexing """
        num_vars = 0
        # Loop over (i,s,j,t)
        # and check if a variable is allowed (nonzero)
        for i in range(len(self.nodes)):
            for s in self.time_points:
                # first check:
                # is s in the time window of i?
                #if s < self.nodes[i].get_window()[0] or s > self.nodes[i].get_window()[1]:
                if not self.check_node_time_compat(i,s):
                    continue
                for j in range(len(self.nodes)):
                    # second check:
                    # is (i,j) an allowed arc?
                    if not self.check_arc((i,j)):
                        continue
                    for t in self.time_points:
                        # third check:
                        # is t in the time window of j?
                        #if t < self.nodes[j].get_window()[0] or t > self.nodes[j].get_window()[1]:
                        if not self.check_node_time_compat(j,t):
                            continue
                        # fourth check
                        # is the travel time from i to j consistent with the timing?
                        if s + self.arcs[(i,j)].get_travel_time() > t:
                            continue

                        # at this point, the tuple (i,s,j,t) is allowed
                        # record the index
                        self.var_mapping.append((i,s,j,t))
                        num_vars += 1
        # end loops
        self.num_variables = num_vars
        self.variables_enumerated = True
        return

    def get_num_variables(self):
        """ number of variables in formulation """
        return self.num_variables

    def get_var_index(self, node_1_index, time_1, node_2_index, time_2):
        """Get the unique id/index of the binary variable given the "tuple" indexing
           Return of None means the tuple does not correspond to a variable
        """
        try:
            return self.var_mapping.index((node_1_index, time_1, node_2_index, time_2))
        except ValueError:
            return None

    def get_var_tuple_index(self, var_index):
        """Inverse of get_var_index"""
        try:
            return self.var_mapping[var_index]
        except IndexError:
            return None

    def build_objective(self):
        """
        Build up linear objective of base BLEC formulation
        """
        if self.objective_built:
           # Objective already built
            return

        self.enumerate_variables()
        # linear terms
        self.objective = np.zeros(self.get_num_variables())
        for k in range(self.get_num_variables()):
            i, _, j, _ = self.var_mapping[k]
            self.objective[k] = self.arcs[(i,j)].get_cost()
        self.objective_built = True
        return

    def build_constraints(self):
        """ Build all constraints of math program """
        if self.constraints_built:
            # Constraints already built
            return
        start = time.time()
        self.build_constraints_quicker()
        #self.build_constraints_exhaustive()
        duration = time.time() - start
        logger.info("Constraint construction took %s seconds", duration)
        return

    def build_constraints_exhaustive(self):
        """
        Build up Linear equality constraints of BLEC
        A*x = b

        SLOW but probably correct? Might add a bunch of vacuous constraints
        """
        self.enumerate_variables()

        aval = []
        arow = []
        acol = []
        brhs = []
        row_index = 0

        # Flow conservation constraints (for each (i,s))
        # EXCEPT DEPOT
        # see above- Depot is first in node list
        for i in range(1,len(self.nodes)):
            for s in self.time_points:
                # sum_jt x_jtis - sum_jt x_isjt = 0
                for j in range(len(self.nodes)):
                    for t in self.time_points:   
                        col = self.get_var_index(j,t,i,s)
                        if col is not None:
                            aval.append(1)
                            arow.append(row_index)
                            acol.append(col)
                        col = self.get_var_index(i,s,j,t)
                        if col is not None:
                            aval.append(-1)
                            arow.append(row_index)
                            acol.append(col)
                # end construction of row entries
                brhs.append(0)
                row_index += 1

        # Sevicing/visitation constraints (for each j)
        # EXCEPT DEPOT (again, depot is first in node list)
        for j in range(1,len(self.nodes)):
            # sum_ist x_isjt = 1
            for i in range(len(self.nodes)):
                for s in self.time_points:
                    for t in self.time_points:
                        col = self.get_var_index(i,s,j,t)
                        if col is not None:
                            aval.append(1)
                            arow.append(row_index)
                            acol.append(col)
            # end construction of row entries
            brhs.append(1)
            row_index += 1

        self.constraints_matrix = sparse.coo_matrix((aval,(arow,acol)))
        self.constraints_rhs = np.array(brhs)
        self.constraints_built = True
        return

    def build_constraints_quicker(self):
        """
        Build up Linear equality constraints of BLEC
        A*x = b

        FASTER
        """
        self.enumerate_variables()

        aval = []
        arow = []
        acol = []
        brhs = []
        row_index = 0

        # Flow conservation constraints (for each (i,s))
        # EXCEPT DEPOT
        # see above- Depot is first in node list
        # First, index the non-trivial constraints
        flow_conservation_mapping = []
        for i in range(1,len(self.nodes)):
            for s in self.time_points:
                # Constraint:
                # sum_jt x_jtis - sum_jt x_isjt = 0

                # is s in the time window of i?
                # (if not, this is a vacuous constraint)
                if s < self.nodes[i].get_window()[0]:
                    continue
                if s > self.nodes[i].get_window()[1]:
                    # time_points is sorted, so s will keep increasing in this loop
                    # This condition will continue to be satisfied
                    # (and so s NOT in time window)
                    break
                flow_conservation_mapping.append((i,s))
                brhs.append(0)
                self.constraint_names.append(f"cflow_{i},{s}")
                row_index += 1
        # NOW, go through variables
        # Note: each variable is an arc, and participates in (at most) TWO constraints:
        # once for INflow to a node, and once for OUTflow from a node
        for col in range(self.get_num_variables()):
            (i,s,j,t) = self.get_var_tuple_index(col)
            # OUTflow:
            try:
                row = flow_conservation_mapping.index((i,s))
                aval.append(-1)
                arow.append(row)
                acol.append(col)
            except ValueError:
                pass
            # INflow:
            try:
                row = flow_conservation_mapping.index((j,t))
                aval.append(1)
                arow.append(row)
                acol.append(col)
            except ValueError:
                pass

        # Servicing/visitation constraints (for each j)
        # EXCEPT DEPOT (again, depot is first in node list)
        # sum_ist x_isjt = 1
        brhs = brhs + [1]*(len(self.nodes) - 1)
        for j in range(1,len(self.nodes)):
            self.constraint_names.append(f"cnode{j}")
        for col in range(self.get_num_variables()):
            (i,s,j,t) = self.get_var_tuple_index(col)
            # We don't care about how many times depot is visited
            if j == 0:
                continue
            aval.append(1)
            arow.append(row_index + (j-1))
            acol.append(col)

#        # Original version (slower)
#        for j in range(1,len(self.nodes)):
#            # sum_ist x_isjt = 1
#            for col in range(self.get_num_variables()):
#                (i,s,jp,t) = self.get_var_tuple_index(col)
#                if jp == j:
#                    aval.append(1)
#                    arow.append(row_index)
#                    acol.append(col)
#            # end construction of row entries
#            brhs.append(1)
#            row_index += 1

        self.constraints_matrix = sparse.coo_matrix((aval,(arow,acol)))
        self.constraints_rhs = np.array(brhs)
        self.constraints_built = True
        return

    def get_arrival_time(self, departure_time, arc):
        """ Get actual arrival time, leaving at <departure_time> on <arc>

        Return:
            arrival_time (float): Arrival time
            in_time_points (bool): whether this arrival time is in time_points set
        """
        # Recall: cannot arrive before time window
        t_w = self.arcs[arc].get_destination().get_window()
        arrival = max(t_w[0], departure_time + self.arcs[arc].get_travel_time())
        greater_time_points = (self.time_points >= arrival)
        if not any(greater_time_points):
            return arrival, False
        arr = np.argmax(greater_time_points)
        arrival_actual = self.time_points[arr]
        return arrival_actual, True

    def make_feasible(self, high_cost):
        """
        Some sort of greedy construction heuristic to make sure the problem is
        feasible. We add dummy nodes/arcs as necessary to emulate more
        vehicles being available.
        """
        # Initialize list of unvisited node indices
        # remove depot
        unvisited_indices = list(range(len(self.nodes)))
        unvisited_indices.remove(0)

        # starting from depot,
        # go through unvisited nodes,
        # greedily visit first unvisited node that we can (satisfying timing constraints)
        # repeat until outgoing arcs from depot are exhausted
        used_arcs = []
        max_vehicles = self.estimate_max_vehicles()
        for _ in range(max_vehicles):
            # start from depot, build a route
            current_node = 0 # depot is first node
            current_time = self.time_points[0]
            building_route = True
            while building_route:
                # go through unvisited nodes, choose first available
                best_node = None
                best_arrival = np.inf
                for n in unvisited_indices:
                    arc = (current_node, n)
                    if self.check_arc(arc):
                        # this is a potentially allowed arc- need to check timing
                        t_w = self.nodes[n].get_window()
                        arrival_actual, in_tp = self.get_arrival_time(current_time, arc)
                        if not in_tp:
                            # arrival time is beyond current time_points
                            continue
                        if arrival_actual <= min(t_w[1], best_arrival):
                            # this timing is valid AND we arrive earlier than all others yet
                            best_node = n
                            best_arrival = arrival_actual
                if best_node is not None:
                    # record arc used, update position + time
                    used_arcs.append((current_node,current_time, best_node,best_arrival))
                    current_node = best_node
                    current_time = best_arrival
                    unvisited_indices.remove(best_node)
                else:
                    # route cannot be continued
                    # Add arc back to depot if possible
                    building_route = False
                    arc = (current_node, 0)
                    assert self.check_arc(arc), f"No arcs back to depot from {current_node}"
                    t_w = self.nodes[0].get_window()
                    arrival_actual, in_tp = self.get_arrival_time(current_time, arc)
                    assert in_tp, f"No arcs back to depot from {current_node} within time horizon"
                    used_arcs.append((current_node, current_time, 0, arrival_actual))
                    # We could potentially add an arc back to depot,
                    # but I think this is messy and an indicator of a malspecified
                    # problem...
                    # self.check_and_add_exit_arc(current_node)
                    # # Make sure that we can get back to depot with the discrete
                    # # time points available
                    # arrival_actual, in_tp = self.get_arrival_time(current_time, arc)
                    # if not in_tp:
                    #     # if arrival time is not in time_points, add it in
                    #     self.time_points = np.append(self.time_points, arrival_actual)
                    # used_arcs.append((current_node,current_time, 0,arrival_actual))
            # end building route
        # end construction over all routes

        # NOW, if there are unvisited nodes, construct expensive dummy arcs from depot
        # and record these dummy routes
        for n in unvisited_indices:
            # Since we are adding arcs, we need to re-construct/enumerate stuff
            self.variables_enumerated = False
            self.constraints_built = False
            self.objective_built = False

            arc = (0, n)
            depot_nm = self.node_names[0]
            node_nm = self.node_names[n]
            assert not self.check_arc(arc), \
                f"We should have been able to construct a route through node {node_nm}"
            logger.info("Adding entry arc to %s", node_nm)
            added = self.add_arc(depot_nm, node_nm, 0, high_cost)
            assert added, "Something is wrong in construction heuristic"
            current_time = self.time_points[0]
            arrival, in_tp = self.get_arrival_time(current_time, arc)
            # Since the arc we added has zero travel time, 
            # arrival time should be in time_points already...
            assert in_tp, f"Arriving at {arrival}: not in time_points??"
            used_arcs.append((0, current_time, n, arrival))
            current_time = arrival

            # Now, exit back to depot
            self.check_and_add_exit_arc(n, high_cost)
            arc = (n, 0)
            arrival, in_tp = self.get_arrival_time(current_time, arc)
            assert in_tp, f"Arriving at {arrival}: not in time_points??"
            used_arcs.append((n, current_time, 0, arrival))
        # done fixing

        # construct and save feasible solution
        self.enumerate_variables()
        self.feasible_solution = np.zeros(self.num_variables)
        for a in used_arcs:
            self.feasible_solution[self.get_var_index(*a)] = 1
        return

    def check_and_add_exit_arc(self, node_index, cost=0):
        """ If exit arc from nodes[node_index] to depot does not exist,
        add it with zero travel time BUT cost of cost
        """
        arc = (node_index, 0)
        if not self.check_arc(arc):
            node_nm = self.node_names[node_index]
            depot_nm = self.node_names[0]
            logger.info("Adding exit arc from %s", node_nm)
            added = self.add_arc(node_nm, depot_nm, 0, cost)
            assert added, f"Something is wrong with exit arcs: {arc} not added"
            # Since we are adding arcs, we need to re-construct/enumerate stuff
            self.variables_enumerated = False
            self.constraints_built = False
            self.objective_built = False
        return

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
        self.build_constraints()
        A_eq = self.constraints_matrix
        b_eq = self.constraints_rhs
        n = self.get_num_variables()
        # if anything is empty, make sure its dense
        if len(b_eq) == 0:
            A_eq = A_eq.toarray()
        return A_eq, b_eq, sparse.csr_matrix((n,n)), 0

    def get_qubo(self, feasibility=False, penalty_parameter=None):
        """
        Get the Quadratic Unconstrained Binary Optimization problem reformulation

        args:
        feasibility (bool): Get the feasibility problem (ignore the objective)
        penalty_parameter (float): value of penalty parameter to use for reformulation.
            If None, it is determined automatically

        Return:
        Q (ndarray): Square matrix defining QUBO
        c (float): a constant that makes the objective of the QUBO equal to the
            objective value of the original constrained integer program
        """
        self.build_objective()
        self.build_constraints()

        if feasibility:
            penalty_parameter = 1.0
        else:
            sum_arc_cost = sum([np.fabs(arc.get_cost()) for arc in self.arcs.values()])
            sufficient_pp = (len(self.time_points)**2)*sum_arc_cost
            if penalty_parameter is None:
                penalty_parameter = sufficient_pp + 1.0
            if penalty_parameter <= sufficient_pp:
                logger.warning(
                    "Penalty parameter might not be big enough...(>%s)", sufficient_pp
                )

        qval = []
        qrow = []
        qcol = []

        # according to scipy.sparse documentation,
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix)
        # Duplicated entries are merely summed to together when converting to an
        # array or other sparse matrix type. This is consistent with our aim

        # Linear objective terms:
        if not feasibility:
            for i in range(self.get_num_variables()):
                if self.objective[i] != 0:
                    qrow.append(i)
                    qcol.append(i)
                    qval.append(self.objective[i])

        # Linear Equality constraints:
        # rho * ||Ax - b||^2 = rho*( x^T (A^T A) x - 2b^T A x + b^T b )

        # Put -2b^T A on the diagonal:
        two_bta = -2*self.constraints_matrix.transpose().dot(self.constraints_rhs)
        for i in range(self.get_num_variables()):
            if two_bta[i] != 0:
                qrow.append(i)
                qcol.append(i)
                qval.append(penalty_parameter*two_bta[i])

        # Construct the QUBO objective matrix so far
        Q = sparse.coo_matrix((qval,(qrow,qcol)),
            shape=(self.get_num_variables(),self.get_num_variables())
        )

        # Add A^T A to it
        # This will be some sparse matrix (probably CSR format)
        Q = Q + penalty_parameter*self.constraints_matrix.transpose().dot(self.constraints_matrix)

        # constant term of QUBO objective
        constant = penalty_parameter*self.constraints_rhs.dot(self.constraints_rhs)

        return Q, constant

    def get_cplex_prob(self):
        """
        Get a CPLEX object containing the original constrained integer program
        representation

        args:
        None

        Return:
        cplex_prob (cplex.Cplex): A CPLEX object defining the MIP problem
        """
        self.build_objective()
        self.build_constraints()

        cplex_prob = cplex.Cplex()

        # Variables: all binary
        # constraints: all equality
        var_types = [cplex_prob.variables.type.binary] * len(self.objective)
        namer = lambda isjt: "n{}t{}_n{}t{}".format(isjt[0], isjt[1], isjt[2], isjt[3])
        names = list(map(namer, self.var_mapping))
        con_types = ['E'] * len(self.constraints_rhs)
        rows = self.constraints_matrix.row.tolist()
        cols = self.constraints_matrix.col.tolist()
        vals = self.constraints_matrix.data.tolist()

        # define object
        cplex_prob.objective.set_sense(cplex_prob.objective.sense.minimize)
        cplex_prob.variables.add(obj=self.objective.tolist(), types=var_types, names=names)
        cplex_prob.linear_constraints.add(rhs=self.constraints_rhs.tolist(), senses=con_types,
            names=self.constraint_names)
        cplex_prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
        return cplex_prob

    def get_routes(self, solution):
        """
        Get a representation of the paths/ vehicle routes in a solution

        solution: binary vector corresponding to a solution
        """
        soln_var_indices = np.nonzero(solution)
        soln_var_indices = soln_var_indices[0]
        # Lexicographically sort the indices; all routes start from Depot (node index zero),
        # so sort the arcs so that those leaving the depot are first
        # Flip the tuples because np.lexsort sorts on last row, second to last row, ...
        soln_var_tuples = [self.get_var_tuple_index(k) for k in soln_var_indices]
        tuples_to_sort = np.flip(np.array(soln_var_tuples), -1)
        arg_sorted = np.lexsort(tuples_to_sort.T)
        tuples_ordered = [soln_var_tuples[i] for i in arg_sorted]
        # While building route, do some dummy checks to make sure formulation is right;
        # check that each node is visited exactly once
        visited = np.zeros(len(self.nodes))
        routes = []
        while len(tuples_ordered) > 0:
            routes.append([])
            # Get first arc not yet in a route
            arc = tuples_ordered.pop(0)
            route_finished = False
            # Follow arcs to build route
            while not route_finished:
                routes[-1].append((arc[0], arc[1]))
                node_to_find = (arc[2], arc[3])
                visited[arc[2]] += 1
                assert self.check_node_time_compat(arc[2], arc[3]), "Node time window not satisfied"
                node_found = False
                for i,a in enumerate(tuples_ordered):
                    if node_to_find == (a[0], a[1]):
                        arc = tuples_ordered.pop(i)
                        node_found = True
                        break
                if not node_found:
                    routes[-1].append(node_to_find)
                    route_finished = True
            # end building route
        # end building all routes
        assert all(visited[1:] == 1), "Solution does not obey node visitation constraints"
        return routes
