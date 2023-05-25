"""
SM Harwood
6 December 2019
"""
import time
import logging
from itertools import product
import numpy as np
from scipy import sparse
from ..vrptw import Arc
from ..routing_problem import RoutingProblem
try:
    import cplex
except ImportError:
    pass

logger = logging.getLogger(__name__)

class SequenceBasedRoutingProblem(RoutingProblem):
    """
    Class to implement the sequence-based formulation of a VRPTW
    Base math program is a 0-1 Integer Quadratically Constrained Quadratic Program
    which can be transformed to a
    Quadratic Unconstrained Binary Optimization (QUBO) problem
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, vrptw=None, strict=True):
        """
        Initialize a sequence-based routing problem.

        Parameters:
        strict (bool): Setting strict = True enforces timing of arcs more strictly
            This may lead to a more physically reasonable solution, while potentially
            causing infeasible problems.
        """
        super().__init__(vrptw)
        self.strict = strict
        self.max_sequence_length = 0
        self.max_vehicles = 0
        self.vehicle_cost = []
        self.num_variables = 0
        self.variables_enumerated = False
        self.var_mapping = []
        self.var_mapping_inverse = None
        self.fixed_values = dict()
        self.objective_built = False
        self.lin_con_built = False
        self.quad_con_built = False
        self.lin_con_names = []

        # Parameters of the constrained math program formulation:
        self.objective_c = None
        self.objective_q = None
        self.quadratic_constraints_matrix = None
        self.linear_constraints_matrix = None
        self.linear_constraints_rhs = None

        # If the underlying VRPTW already has arcs, we should check that they
        # satisfy the conditions implied by `strict`
        if strict:
            old_arcs = self.arcs
            self.vrptw.arcs = dict()
            for arc in old_arcs.values():
                self.add_arc(
                    arc.origin.name,
                    arc.destination.name,
                    arc.travel_time,
                    arc.cost
                )
        # Also, if there is a depot, use the derived class set_depot
        try:
            depot_nm = self.node_names[self.depot_index]
            self.set_depot(depot_nm)
        except IndexError:
            pass

    def set_max_sequence_length(self, max_sequence_length):
        """ Set the maximum length of a route/sequence of moves """
        self.max_sequence_length = int(max_sequence_length)
        return

    def set_max_vehicles(self, max_vehicles):
        """
        Set the maximum number of vehicles available
        May be less than self.estimate_max_vehicles()
        """
        self.max_vehicles = max_vehicles
        self.vehicle_cost = [0]*max_vehicles
        return

    def set_depot(self, depot_name):
        """Take node named `depot_name` and move to first position in list"""
        super().set_depot(depot_name)
        # Since we treat the depot as absorbing
        # (see build_objective_quadratic_constraints)
        # We should allow Depot - Depot moves
        self.arcs[(0,0)] = Arc(self.nodes[0], self.nodes[0], 0, 0)
        return

    def add_arc(self, origin_name, destination_name, travel_time, cost=0):
        """
        Add a potentially allowable arc;
        we also check feasibility of TIMING:
            allow if origin time window START plus travel time <= destination time window end
            (There is no way this arc could ever be used if this is not satisfied)
        For this formulation we can also enforce timing more strictly:
            allow if origin time window END   plus travel time <= destination time window end
            (unless the origin is the depot)

        Return:
            added (bool): whether arc was added or not
        """
        i = self.get_node_index(origin_name)
        if self.strict and i != 0:
            j = self.get_node_index(destination_name)
            departure_time = self.nodes[i].get_window()[1]
            if departure_time + travel_time <= self.nodes[j].get_window()[1]:
                self.arcs[(i,j)] = Arc(self.nodes[i], self.nodes[j], travel_time, cost)
                return True
            # else
            return False
        # else not the strict formulation, or the origin node is the depot -
        # follow the base class logic
        return super().add_arc(origin_name, destination_name, travel_time, cost)

    def check_arc(self, arc_key):
        """ Is the arc valid? """
        return arc_key in self.arcs

    def enumerate_variables(self):
        """
        Basic operation that needs to be done to keep track of variable counts,
        indexing
        """
        if self.variables_enumerated:
            return

        # "inverse" var map - from tuple index to enumerated index
        # use -1 to indicate the var is fixed
        self.var_mapping_inverse = -np.ones(
            (self.max_vehicles, self.max_sequence_length, len(self.nodes)),
            dtype=int
        )
        start = time.time()
        num_vars = 0
        # Loop over (vehicles, positions/sequence, nodes)
        # and check if a variable is free/not fixed
        for si in range(self.max_sequence_length):
            for ni in range(len(self.nodes)):
                # Is the variable (vi,si,ni) fixed?
                # Check the rules:
                # We must start in the depot:
                if si == 0 and ni == 0:
                    for vi in range(self.max_vehicles):
                        self.fixed_values[(vi,si,ni)] = 1.0
                    continue
                # We must NOT start anywhere besides depot:
                if si == 0 and ni != 0:
                    for vi in range(self.max_vehicles):
                        self.fixed_values[(vi,si,ni)] = 0.0
                    continue
                # We must follow allowed edges from depot
                if si == 1 and not self.check_arc((0,ni)):
                    for vi in range(self.max_vehicles):
                        self.fixed_values[(vi,si,ni)] = 0.0
                    continue
                # We must END in the depot:
                if si == self.max_sequence_length-1 and ni == 0:
                    for vi in range(self.max_vehicles):
                        self.fixed_values[(vi,si,ni)] = 1.0
                    continue
                # We must NOT end anywhere besides depot:
                if si == self.max_sequence_length-1 and ni != 0:
                    for vi in range(self.max_vehicles):
                        self.fixed_values[(vi,si,ni)] = 0.0
                    continue
                # We must follow allowed edges back to depot
                if si == self.max_sequence_length-2 and not self.check_arc((ni,0)):
                    for vi in range(self.max_vehicles):
                        self.fixed_values[(vi,si,ni)] = 0.0
                    continue
                # At this point, variable (vi,si,ni) is free, record the index
                for vi in range(self.max_vehicles):
                    self.var_mapping.append((vi,si,ni))
                    self.var_mapping_inverse[vi,si,ni] = num_vars
                    num_vars += 1
        # end loops
        duration = time.time() - start
        logger.info("Variable enumeration took %s seconds", duration)
        self.num_variables = num_vars
        self.variables_enumerated = True
        return

    def get_num_variables(self):
        """ Number of variables in formulation """
        if not self.variables_enumerated:
            self.enumerate_variables()
        return self.num_variables

    def get_var_index(self, vehicle_index, sequence_index, node_index):
        """
        Get the unique id/index of the binary variable given the "tuple" indexing
        Return of None means the tuple corresponds to a fixed variable
        """
        index = self.var_mapping_inverse[vehicle_index, sequence_index, node_index]
        if index < 0:
            return None
        # else
        return index

    def get_var_tuple_index(self, var_index):
        """Inverse of get_var_index"""
        try:
            return self.var_mapping[var_index]
        except IndexError:
            return None

    def build_objective(self):
        """
        Build up objective of base constrained math program:

        objective_q: quadratic/bilinear coefficients
        objective_c: linear coefficients
        """
        if self.objective_built:
            return
        self.enumerate_variables()

        # linear terms of objective
        self.objective_c = np.zeros(self.get_num_variables())
        # quadratic (bilinear) terms of objective
        qval = []
        qrow = []
        qcol = []

        # Linear and Bilinear terms
        for vi in range(self.max_vehicles):
            for si in range(self.max_sequence_length-1):
                for (ni,nj) in self.arcs.keys():
                    key = (ni,nj)
                    var_index_1 = self.get_var_index(vi,si,ni)
                    var_index_2 = self.get_var_index(vi,si+1,nj)

                    # if both variables fixed, this goes to constant part of obj
                    # if exactly one is fixed, this goes to linear part
                    # if no variable is fixed, this goes to quadratic part
                    constant_bool = ((var_index_1 is None) and (var_index_2 is None))
                    linear_bool   = ((var_index_1 is None) !=  (var_index_2 is None)) # exclusive or
                    quad_bool = not ((var_index_1 is None) or  (var_index_2 is None))

                    coeff = self.arcs[key].get_cost() + self.vehicle_cost[vi]
                    if constant_bool:
                        constant = self.fixed_values[(vi,si,ni)]*self.fixed_values[(vi,si+1,nj)]
                        if constant != 0:
                            print("Kind of weird... not gonna track constant part of objective")
                    if linear_bool:
                        if var_index_1 is None:
                            coeff *= self.fixed_values[(vi,si,ni)]
                            var_index = var_index_2
                        if var_index_2 is None:
                            coeff *= self.fixed_values[(vi,si+1,nj)]
                            var_index = var_index_1
                        self.objective_c[var_index] += coeff
                    if quad_bool:
                        qrow.append(var_index_1)
                        qcol.append(var_index_2)
                        qval.append(coeff)
        # construct sparse matrix for bilinear terms
        M = self.get_num_variables()
        self.objective_q = sparse.coo_array((qval,(qrow,qcol)), shape=(M,M))
        self.objective_built = True
        return

    def build_quadratic_constraints(self):
        """
        Build up quadratic (bilinear) constraints of base math program
        """
        if self.quad_con_built:
            return
        self.enumerate_variables()

        # Certain arcs cannot be used.
        # These arcs must be penalized in the objective.
        # Keep track with quadratic_constraints_matrix

        start = time.time()
        # for bilinear constraint/penalty terms
        pqrow = []
        pqcol = []

        # Constraint: Only use allowed arcs
        # x_{vi,si,ni} * x_{vi,si+1,nj} = 0, \forall vi,si,(ni,nj) \notin arcs
        for ni, nj in product(range(len(self.nodes)),range(len(self.nodes))):
            # If this is a valid arc, then there is no constraint
            if self.check_arc((ni,nj)):
                continue
            for si in range(self.max_sequence_length-1):
                for vi in range(self.max_vehicles):
                    pqrow, pqcol = self.quadratic_constraint_logic(
                        vi, si, ni, nj, pqrow, pqcol
                    )

        # Constraint: Once a vehicle returns to depot, it remains there
        # x_{vi,si,d}  * x_{vi,si+1,nj} = 0, \forall vi, si >= 1, nj \neq d
        for vi in range(self.max_vehicles):
            for si in range(1,self.max_sequence_length-1):
                for nj in range(1,len(self.nodes)):
                    # If this is not a valid arc, then we already added this constraint above
                    if not self.check_arc((0,nj)):
                        continue
                    pqrow, pqcol = self.quadratic_constraint_logic(
                        vi, si, 0, nj, pqrow, pqcol
                    )

        # construct sparse matrix for bilinear terms
        n_var = self.get_num_variables()
        pqval = np.ones(len(pqrow))
        self.quadratic_constraints_matrix = sparse.coo_array((pqval,(pqrow,pqcol)),
            shape=(n_var,n_var)
        )
        self.quad_con_built = True
        duration = time.time() - start
        logger.info("Quadratic constraints built in %s seconds", duration)
        return

    def quadratic_constraint_logic(self, vi, si, ni, nj, pqrow, pqcol):
        """
        Consistently check the quadratic constraints and update sparse
        representation of the matrix
        """
        var_index_1 = self.get_var_index(vi,si,ni)
        var_index_2 = self.get_var_index(vi,si+1,nj)

        # # if one or both variables fixed, check consistency
        # # if no variable is fixed, this goes to quadratic constraints
        # constant_bool = ((var_index_1 is None) and (var_index_2 is None))
        # linear_bool   = ((var_index_1 is None) !=  (var_index_2 is None)) # exclusive or
        # quad_bool = not ((var_index_1 is None) or  (var_index_2 is None))

        # BOTH VARS FIXED
        if (var_index_1 is None) and (var_index_2 is None):
            # constraint is x_i * x_j = 0
            fixed_val = self.fixed_values[(vi,si,ni)]* \
                        self.fixed_values[(vi,si+1,nj)]
            assert np.isclose(fixed_val, 0.0), \
                "Quadratic constraint not consistent"
        # ONE AND ONLY ONE VAR FIXED (exclusive or)
        elif (var_index_1 is None) != (var_index_2 is None):
            # constraint is x_i * x_j = 0
            # If only one value is fixed, it must be zero,
            # otherwise we have missed a chance to fix a variable
            if var_index_1 is None:
                fixed_val = self.fixed_values[(vi,si,ni)]
                missed_var_index = (vi,si+1,nj)
            else:
                fixed_val = self.fixed_values[(vi,si+1,nj)]
                missed_var_index = (vi,si,ni)
            assert np.isclose(fixed_val, 0.0), \
                f"Missed chance to fix variable {missed_var_index}"
        # NEITHER VAR FIXED
        else:
            # Either invalid arc or enforcing depot absorption
            # Update quadratic constraint/penalty
            pqrow.append(var_index_1)
            pqcol.append(var_index_2)
        return pqrow, pqcol

    def build_linear_constraints(self):
        """
        Build up Linear equality constraints of base math program
        A*x = b
        """
        if self.lin_con_built:
            return
        self.enumerate_variables()

        start = time.time()
        aval = []
        arow = []
        acol = []
        brhs = []

        row_index = 0
        # Each node (except depot) is visited exactly once
        for ni in range(1,len(self.nodes)):
            # right-hand side value is one
            brhs.append(1.0)
            self.lin_con_names.append(f"c_node{ni}")
            # sum over sequence and vehicle indices
            for si in range(self.max_sequence_length):
                for vi in range(self.max_vehicles):
                    var_index = self.get_var_index(vi,si,ni)
                    if var_index is None:
                        # Fixed variable.
                        # "move" it to right-hand side
                        brhs[-1] -= self.fixed_values[(vi,si,ni)]
                        continue
                    arow.append(row_index)
                    acol.append(var_index)
                    aval.append(1.0)
                # end for
            # end for
            row_index += 1

        # For each vehicle, each sequence index is used exactly once
        # The first and last sequence positions are automatically satisfed by fixed variable values
        for si in range(1,self.max_sequence_length-1):
            for vi in range(self.max_vehicles):
                # right-hand side value is one
                brhs.append(1.0)
                self.lin_con_names.append(f"c_v{vi}s{si}")
                # sum over all nodes
                for ni in range(len(self.nodes)):
                    var_index = self.get_var_index(vi,si,ni)
                    if var_index is None:
                        # Fixed variable.
                        # "move" it to right-hand side
                        brhs[-1] -= self.fixed_values[(vi,si,ni)]
                        continue
                    arow.append(row_index)
                    acol.append(var_index)
                    aval.append(1.0)
                # end for
                row_index += 1

        self.linear_constraints_matrix = sparse.coo_array((aval,(arow,acol)))
        self.linear_constraints_rhs = np.array(brhs)
        self.lin_con_built = True
        duration = time.time() - start
        logger.info("Linear constraints built in %s seconds", duration)
        return

    def make_feasible(self, high_cost):
        """
        Some sort of greedy construction heuristic to make sure the problem is
        feasible. We add dummy nodes/arcs as necessary to emulate more
        vehicles being available.
        """
        # Initialize list of unvisited node indices
        # remove depot
        # then sort based on time window end - useful later
        unvisited_indices = list(range(len(self.nodes)))
        unvisited_indices.remove(0)
        unvisited_indices.sort(key=lambda n: self.nodes[n].get_window()[1])

        depot_nm = self.node_names[0]
        used_sequences = []
        for vi in range(self.max_vehicles):
            # We always start and end in depot; the corresponding variables are fixed
            current_node = 0
            for si in range(1,self.max_sequence_length-1):
                # Although timing is less important in this formulation,
                # we still choose the next node based on the earliest next departure time,
                # as this will be a proxy for how much flexibility/ how many more
                # nodes we can visit on this route
                no_next_node = True
                for ni in unvisited_indices:
                    arc = (current_node, ni)
                    if self.check_arc(arc):
                        # unvisited_indices is sorted, so the first node to which
                        # we have a valid arc is the node with best timing
                        used_sequences.append((vi, si, ni))
                        unvisited_indices.remove(ni)
                        current_node = ni
                        no_next_node = False
                        break
                if no_next_node:
                    # We can't go to any unvisited nodes
                    # finish out the route at the depot
                    arc = (current_node, 0)
                    if not self.check_arc(arc):
                        node_nm = self.node_names[current_node]
                        self.add_arc(node_nm, depot_nm, 0, 0)
                        logger.info("Adding arc %s -- %s", node_nm, depot_nm)
                    for sii in range(si, self.max_sequence_length-1):
                        used_sequences.append((vi, sii, 0))
                    break
            # end sequence loop
        # end vehicle loop

        for ni in unvisited_indices:
            # We are changing data - variables counts will change
            self.variables_enumerated = False
            self.objective_built = False
            self.lin_con_built = False
            self.quad_con_built = False

            # add a vehicle and entry arc from depot
            # arcs from the depot could be "available"
            # So add a cost for this dummy *vehicle*
            vi = self.max_vehicles
            self.max_vehicles += 1
            self.vehicle_cost.append(high_cost)
            logger.info("Adding vehicle %s with cost %s", vi, high_cost)
            arc = (0, ni)
            node_nm = self.node_names[ni]
            # check and add entry arc
            if not self.check_arc((0, ni)):
                self.add_arc(depot_nm, node_nm, 0, high_cost)
                logger.info("Adding arc %s -- %s", depot_nm, node_nm)
            used_sequences.append((vi, 1, ni))
            # check and add exit arc
            if not self.check_arc((ni, 0)):
                self.add_arc(node_nm, depot_nm, 0, high_cost)
                logger.info("Adding arc %s-- %s", node_nm, depot_nm)
            # finish out sequence at depot
            for si in range(2, self.max_sequence_length-1):
                used_sequences.append((vi, si, 0))
        # end modifying problem to make feasible

        # construct and save feasible solution
        self.enumerate_variables()
        self.feasible_solution = np.zeros(self.num_variables)
        for seq in used_sequences:
            self.feasible_solution[self.get_var_index(*seq)] = 1
        return

    def get_objective_data(self):
        """
        Return objective information in a consistent way
        objective(x) = cᵀx + xᵀ Q x

        Parameters:

        Return:
            c (array): 1-d array defining linear part of objective
            Q (array): 2-d array defining quadratic part of objective
        """
        self.build_objective()
        return self.objective_c, self.objective_q

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
        self.build_linear_constraints()
        self.build_quadratic_constraints()
        A_eq = self.linear_constraints_matrix
        b_eq = self.linear_constraints_rhs
        # Linearizing bilinear constraints is just too big
        Q_eq = sparse.csr_array(self.quadratic_constraints_matrix)
        r_eq = 0
        # if anything is empty, make sure its dense
        if len(b_eq) == 0:
            A_eq = A_eq.toarray()
        return A_eq, b_eq, Q_eq, r_eq

    def get_sufficient_penalty(self, feasibility):
        """
        Return a threshhold value of the penalty parameter that is sufficient
        for penalizing the constraints when constructing a QUBO representation of
        this problem

        Parameters:
            feasibility (bool): Whether this is for a feasibility version of the
                problem. Sufficient penalty value can be zero

        Return:
            sufficient_pp (float): Penalty parameter value
        """
        if feasibility:
            sufficient_pp = 0.0
        else:
            sum_arc_cost = sum(np.fabs(arc.get_cost()) for arc in self.arcs.values())
            sufficient_pp = self.max_sequence_length*self.max_vehicles*sum_arc_cost
        return sufficient_pp

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
        self.build_linear_constraints()
        self.build_quadratic_constraints()

        # Define object
        cplex_prob = cplex.Cplex()
        cplex_prob.objective.set_sense(cplex_prob.objective.sense.minimize)
        # Variables: all binary
        # can't go too wild with names otherwise .lp format doesn't like it
        namer = lambda t: "v{}s{}n{}".format(t[0], t[1], t[2])
        names = list(map(namer, self.var_mapping))
        var_types = [cplex_prob.variables.type.binary] * self.get_num_variables()
        cplex_prob.variables.add(obj=self.objective_c.tolist(), types=var_types, names=names)

        # Linear constraints: all equality
        lcon_types = ['E'] * len(self.linear_constraints_rhs)
        rows = self.linear_constraints_matrix.row.tolist()
        cols = self.linear_constraints_matrix.col.tolist()
        vals = self.linear_constraints_matrix.data.tolist()
        cplex_prob.linear_constraints.add(
            rhs=self.linear_constraints_rhs.tolist(),
            senses=lcon_types, 
            names=self.lin_con_names
        )
        cplex_prob.linear_constraints.set_coefficients(zip(rows, cols, vals))

        # Bilinear constraints: LINEARIZE
        # x_i * x_j = 0 for certain i,j
        # <==>
        # x_i + x_j <= 1 (when vars are binary)
        rows = self.quadratic_constraints_matrix.row
        cols = self.quadratic_constraints_matrix.col
        vals = self.quadratic_constraints_matrix.data
        assert (vals == 1.0).all(), "Linearization plan not gonna work"
        linearized = [cplex.SparsePair(ind = [int(r), int(c)], val = [1.0, 1.0])
            for r,c in zip(rows, cols)]
        names = [f"c_linearized_{r}_{c}" for r,c in zip(rows, cols)]
        num_to_add = len(linearized)
        cplex_prob.linear_constraints.add(
            lin_expr=linearized,
            senses=['L']*num_to_add,
            rhs=[1.0]*num_to_add,
            names=names)

        # Quadratic objective
        # Divide values by 2? CPLEX claims it adds the values in symmetrically...
        rows = self.objective_q.row.tolist()
        cols = self.objective_q.col.tolist()
        vals = self.objective_q.data.tolist()
        cplex_prob.objective.set_quadratic_coefficients(zip(rows,cols,vals))

#        # Quadratic objective PLUS quadratic constraints as penalty:
#        pen_param = 1.0 + self.get_sufficient_penalty(feasibility=False)
#        complete_q = (self.objective_q + pen_param*self.quadratic_constraints_matrix).tocoo()
#        rows = complete_q.row.tolist()
#        cols = complete_q.col.tolist()
#        vals = complete_q.data.tolist()
#        cplex_prob.objective.set_quadratic_coefficients(zip(rows,cols,vals))
        return cplex_prob

    def get_routes(self, solution):
        """
        Get a representation of the paths/ vehicle routes in a solution

        solution: binary vector corresponding to a solution
        """
        soln_var_indices = np.flatnonzero(solution)
        if soln_var_indices.size == 0:
            logger.info("A strange game. The only winning move is not to play")
            return []
        # The indices are (vehicle, position, node)
        # If we lexicographically sort them we automatically get the routes for each vehicle
        # Add in the fixed values (enforcing that vehicles start at depot)
        # Flip the tuples because np.lexsort sorts on last row, second to last row, ...
        soln_var_tuples = [self.get_var_tuple_index(k) for k in soln_var_indices]
        soln_var_tuples += [t for t,v in self.fixed_values.items() if v == 1.0]
        tuples_to_sort = np.flip(np.array(soln_var_tuples), -1)
        arg_sorted = np.lexsort(tuples_to_sort.T)
        tuples_ordered = [soln_var_tuples[i] for i in arg_sorted]
        logger.debug("tuples_ordered=%s", tuples_ordered)

        routes = []
        # Build up routes and do dummy checks
        for vi in range(self.max_vehicles):
            routes.append([])
            prev_node = None
            for si in range(self.max_sequence_length):
                t = tuples_ordered.pop(0)
                curr_node = t[2]
                if t[0] != vi or t[1] != si:
                    logger.warning("Unexpected tuple %s in solution", t)
                    continue
                if prev_node and not self.check_arc((prev_node, curr_node)):
                    logger.warning("Solution uses unallowed arc %s - %s", prev_node, curr_node)
                    continue
                routes[-1].append(curr_node)
                prev_node = curr_node
        return routes

    # def print_objective(self):
    #     """Just display the sparse matrix encoding linear/bilinear terms in a nice way"""
    #     print("Linear terms:")
    #     for i in range(self.get_num_variables()):
    #         (vi,si,ni) = self.get_var_tuple_index(i)
    #         print("v{}, s{}, {}: {}".format(vi,si,self.node_names[ni], self.objective_c[i]))

    #     print("Quadratic terms:")
    #     for (r,c,val) in zip(self.objective_q.row,self.objective_q.col,self.objective_q.data):
    #         (vi,si,ni) = self.get_var_tuple_index(r)
    #         (vj,sj,nj) = self.get_var_tuple_index(c)
    #         print("(v{}, s{}, {}) -- (v{}, s{}, {}) : {}".format(
    #                 vi,si,self.node_names[ni], vj,sj,self.node_names[nj], val))

    # def print_edge_penalty(self):
    #     """
    #     Just display the sparse matrix encoding the edge penalty bi/linear
    #     terms in a nice way
    #     """
    #     print("Quadratic Edge Penalty terms:")
    #     for (r,c,val) in zip(self.quadratic_constraints_matrix.row,
    #                          self.quadratic_constraints_matrix.col,
    #                          self.quadratic_constraints_matrix.data):
    #         (vi,si,ni) = self.get_var_tuple_index(r)
    #         (vj,sj,nj) = self.get_var_tuple_index(c)
    #         print("(v{}, s{}, {}) -- (v{}, s{}, {}) : (not allowed)".format(
    #                 vi,si,self.node_names[ni], vj,sj,self.node_names[nj]))
