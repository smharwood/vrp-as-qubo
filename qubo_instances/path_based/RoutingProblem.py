# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 09:42:01 2018

@author: stuart.m.harwood@exxonmobil.com

Defining a routing problem
Specifically, as seen as a set partitioning problem.
See Desrochers, Desrosiers, Solomon, "A new optimization algorithm for the vehicle routing problem with time windows"

The ultimate goal is to express an instance as a Quadratic Unconstrained Binary Optimization problem
"""
import sys, logging
import datetime, numpy
import scipy.sparse as sparse
from scipy.special import softmax
from itertools import repeat
#import cplex

logger = logging.getLogger(__name__)


class RoutingProblem:
    """
    The Routing Problem, as a Binary Linear Equality-Constrained (BLEC) program
    has the following formulation:

    \min_x \sum_r c_r x_r
      s.t.
      \sum_r \delta_k,r x_r = 1, for each node k,
      x_r \in {0,1}, for all routes r

    where
        x_r = 1 if route r is chosen (0 otherwise)
        c_r = cost of route r
        \delta_k,r = 1 if route r visits node k

    The interpretation is that we wish to choose vehicle routes,
    so that each (non-depot) node is visited exactly once,
    while minimizing the total cost of traveling those routes.

    A valid route begins and ends at the "depot" node,
    and visits other nodes at most once while traversing available arcs.
    In addition, there are demand and delivery time window constraints that
    are encoded in what is a valid or feasible route
    """

    def __init__(self):
        # basic data
        self.NodeNames = []
        self.Nodes = []
        self.Arcs = dict()
        self.depotIndex = 0  # default depot is zeroth node
        self.vehicleCap = 0
        self.initial_loading = 0
        self.routes = []
        self.route_costs = []
        self.route_node_visited = []
        # Parameters of the base formulation:
        # Binary Linear Equality-Constrained (BLEC) problem
        # Generate on the fly from the above data

    def getNodeIndex(self, NodeName):
        return self.NodeNames.index(NodeName)

    def getNode(self, NodeName):
        return self.Nodes[self.getNodeIndex(NodeName)]

    def setDepot(self, DepotName):
        self.depotIndex = self.NodeNames.index(DepotName)
        return

    def setVehicleCap(self, VehicleCap):
        self.vehicleCap = VehicleCap
        return

    def setInitialLoading(self, Loading):
        self.initial_loading = Loading
        return

    def addNode(self, NodeName, Demand, TW=(0, numpy.inf)):
        assert NodeName not in self.NodeNames, NodeName + ' is already in Node List'
        self.Nodes.append(Node(NodeName, Demand, TW))
        self.NodeNames.append(NodeName)
        return

    def addArc(self, OName, DName, Time, Cost=0):
        i = self.getNodeIndex(OName)
        j = self.getNodeIndex(DName)
        self.Arcs[(i, j)] = Arc(self.Nodes[i], self.Nodes[j], Time, Cost)
        return

    def getNumVariables(self):
        return len(self.route_costs)

    def checkRoute(self, CandidateRoute):
        """ Check to see if this path is an allowed route; does it use valid arcs, visit nodes
        within allowed time windows, and satisfy cumulative demand

        args:
        CandidateRoute (list of int or list of strings): A route to potentially add, defined either
            as a list of node indexes or node names

        Return:
        feasible (bool): whether the route is a viable route
        cost (float): associated cost of route
        visitsNode (list of int): List indicating whether
            this route visits Node[i] (visitsNode[i] = 1)
            or not (visitsNode[i] = 0)
        """
        feasible = True
        cost = 0
        visitsNode = [0] * (len(self.Nodes))
        if len(CandidateRoute) < 2:
            # must be at least 2 stops long
            return False, cost, visitsNode

        RouteIndices = CandidateRoute
        # Convert from list of string node names to indices if necessary
        # (rest are done on the fly)
        check_indexes = (0, 1, -1)
        for index in check_indexes:
            if isinstance(RouteIndices[index], str):
                RouteIndices[index] = self.getNodeIndex(RouteIndices[index])

        # First check:
        # are first and last nodes the depot?
        if RouteIndices[0] != self.depotIndex or RouteIndices[-1] != self.depotIndex:
            return False, cost, visitsNode
        # Make sure these are valid arcs,
        # visit each node at most once (besides depot)
        # satisfy capacity constraints
        # and time window constraints
        time = 0
        loading = self.initial_loading
        for i in range(len(RouteIndices) - 1):
            # have we already visited this node?
            if visitsNode[RouteIndices[i]] == 1:
                return False, cost, visitsNode
            else:
                visitsNode[RouteIndices[i]] = 1

            # valid arc?
            # first, convert to index
            if isinstance(RouteIndices[i + 1], str):
                RouteIndices[i + 1] = self.getNodeIndex(RouteIndices[i + 1])
            a = (RouteIndices[i], RouteIndices[i + 1])
            # will increment time and loading appropriately
            feasArc, time, loading = self.checkArc(time, loading, a)
            if feasArc:
                cost += self.Arcs[a].getCost()
            else:
                return False, cost, visitsNode
        # end for loop
        return feasible, cost, visitsNode

    def checkArc(self, time, load, arcKey):
        """ Check if an arc is part of a valid route

        args:
        time (float): Time accumulated so far on route (time leaving origin of arc)
        load (float): Load on vehicle
        arcKey (tuple of int): The pair of node indices of potential arc

        return:
        feasibleArc (bool): Whether arc is feasible/part of valid route
        time (float): Updated time/arrival at destination of arc
        load (float): Updated load
        """
        # is the key even valid?
        try:
            arc = self.Arcs[arcKey]
            dest = arc.getD()
        except(KeyError):
            return False, time, load
        # Do we arrive before or during the next node's time window?
        # add travel time of current arc,
        # but if we arrive at a node before its time window, we have to wait
        time += arc.getTravelTime()
        time = max(time, dest.getWindow()[0])
        if time > dest.getWindow()[1]:
            return False, time, load
        # Is the load physical (nonnegative and within capacity)?
        load += dest.getLoad()
        if load > self.vehicleCap or load < 0:
            return False, time, load
        return True, time, load

    def getRouteNames(self, route):
        """ Given a sequence of indices, return the corresponding node names """
        return list(map(lambda n: self.NodeNames[n], route))

    def generateRoute(self, vf=None, explore=1, node_costs=None, time_costs=None, unvisited=None):
        """ Generate a route
        Can view this as one iteration of an approximate dynamic programming method
        The goal is to find a good candidate route, and we always have exploration,
            so it is not "true" DP to find an optimal route

        args:
        vf (list-like of float): value function over nodes
        explore (float): Parameter controlling exploration/sampling;
            explore=0 : sample tightly around mode
            explore=np.inf : uniform sampling over keys
        node_costs (list-like of float): Cost of visiting a node (to add to stage costs)
            Zero if node_costs is None
        time_costs (function): Cost of arrival time at a node; takes a float and returns a float
        unvisited (list): List of unvisited node indices. Generated route will only contain nodes
            from this list. If None, all nodes are possible to visit

        Return:
        r (list of int): A route
        vf (list of float): updated (in place) value function
        """

        if vf is None:
            vf = [0] * len(self.Nodes)
        else:
            assert len(vf) == len(self.Nodes), 'Value function incorrect size'

        if unvisited is None:
            unvisited = list(range(len(self.Nodes)))

        # all routes start at depot
        currNode = self.depotIndex
        r = [currNode]
        time = 0
        load = self.initial_loading
        maxLegs = 2 + len(self.Nodes)
        # Build up a route
        for _ in range(maxLegs):
            PotentialNodesAndVals = dict()
            # Loop over arcs to unvisited nodes,
            # get stage cost plus value function at each node
            for n in unvisited:
                nc = 0.0 if node_costs is None else node_costs[n]
                a = (currNode, n)
                # valid arc?
                feasArc, new_time, _ = self.checkArc(time, load, a)
                tc = 0.0 if time_costs is None else time_costs(new_time)
                if feasArc:
                    stageCost = self.Arcs[a].getCost()
                    PotentialNodesAndVals[n] = stageCost + nc + tc + vf[n]
                else:
                    continue
            # end for

            # Get a node to go to;
            # Minimize value function
            # Maybe add some randomization, proportional to this objective
            try:
                sampledNode, minNode = getSampledKey(PotentialNodesAndVals, explore)
                a = (currNode, sampledNode)
                feasArc, time, load = self.checkArc(time, load, a)
                # Update value function estimate:
                # ACTUAL minimizing cost-to-go: stageCost(a) + vf[minNode]
                vf[currNode] = PotentialNodesAndVals[minNode]
                currNode = sampledNode
            except AssertionError:
                # this probably shouldn't happen
                break
            r.append(currNode)
            # if we have returned to the depot, we are done
            if currNode == self.depotIndex:
                break
        # end loop over building up route
        return r, vf

    def addRoute(self, route):
        """ If route is feasible, save its data

        args:
        route (list of int or list of string): A route to potentially add, defined either as a list
            of node indexes or node names

        Return:
        added (bool): whether the route was added
        cost (float): associated cost of route
        visitsNode (list of int): List indicating whether
            this route visits Node[i] (visitsNode[i] = 1)
            or not (visitsNode[i] = 0)
        """
        # If route r is feasible:
        #  cost = c_r
        #  visitsNode[k] = \delta_k,r
        feas, cost, visitsNode = self.checkRoute(route)
        added = False
        if feas and route not in self.routes:
            self.routes.append(list(route))
            self.route_costs.append(cost)
            self.route_node_visited.append(visitsNode)
            added = True
        # return a copy of visitsNode so original data is not accidentally changed
        return added, cost, list(visitsNode)

    def estimate_max_vehicles(self):
        """ Estimate maximum number of vehicles based on degree of depot node in graph """
        num_outgoing = 0
        num_incoming = 0
        for arc in self.Arcs.keys():
            if arc[0] == self.depotIndex:
                num_outgoing += 1
            if arc[1] == self.depotIndex:
                num_incoming += 1
        return min(num_outgoing, num_incoming)

    def addRoutes_better(self, explore, node_costs, time_costs):
        """ Add routes in a more constructive way

        args:
        explore (float): Parameter controlling exploration/sampling;
            explore=0 : sample tightly around mode
            explore=np.inf : uniform sampling over keys
        node_costs (list-like of float): Cost of visiting a node (to add to stage costs)
            Zero if node_costs is None
        time_costs (function): Cost of arrival time at a node; takes a float and returns a float

        return:
        None
        """
        num_vehicles = self.estimate_max_vehicles()
        unvisited_indices = list(range(len(self.Nodes)))

        # The number of routes that can be added is limited by the number of vehicles
        for _ in range(num_vehicles):
            r, _ = self.generateRoute(None, explore, node_costs, time_costs, unvisited_indices)
            self.addRoute(r)
            # update unvisited indices
            for n in r:
                if n == self.depotIndex:
                    continue
                unvisited_indices.remove(n)
        # end loop
        return

    def addRoutes(self, add_rate=0.5, explore=0, vf=None):
        """ Generate and add a batch of routes to problem

        Args:
        add_rate (float): Parameter controlling how many routes to add; keep adding routes until the
            number of routes that have actually been added divided by the number of attempts is less
            than add_rate
        explore (float): Parameter controlling exploration/sampling;
            explore=0 : sample tightly around mode
            explore=np.inf : uniform sampling
        vf (array-like): value function over nodes

        Return:
        vf (array-like): Updated value function over nodes
        """
        # Value function
        if vf is None:
            vf = numpy.zeros(len(self.Nodes))
        else:
            vf = numpy.asarray(vf)
        assert len(vf) == len(self.Nodes), 'Value function incorrect size'

        # Generate a route with some random exploration
        num_added = 1
        total_tried = 1
        log_iter = 100
        while num_added / total_tried >= add_rate:
            if total_tried % log_iter == 0:
                logger.info("Add rate: {}".format(num_added / total_tried))
            # vf is updated in place by generateRoute
            r, vf = self.generateRoute(vf, explore)
            added, _, _ = self.addRoute(r)
            total_tried += 1
            if added: num_added += 1
            NodesVisited = numpy.sum(self.route_node_visited, axis=0).tolist()

        logger.info("Number of routes added: {}".format(num_added - 1))
        if all(NodesVisited):
            logger.info('All nodes covered!')
        else:
            logger.info('Some node not covered by a route')
        return vf

    def make_feasible(self, cplex_prob, high_cost=None):
        """ Add dummy variables to a Cplex problem to make it automatically feasible

        One variable for each node is added; trivially make problem feasible
        (essentially like phase I of simplex -
         concatenate an identity matrix onto constraint matrix)
        Need to give high cost so it is favorable to NOT use them
        If we focus on feasibility (and all variables have cost zero), we can just set this to one

        args:
        cplex_prob (cplex.Cplex object): MIP problem to modify
        high_cost (float): "High cost" to use to penalize dummy variables.
            if None, it is automatically calculated based on maximum arc cost and maximum number of
            nodes in a route

        returns:
        cplex_prob: Modified object
        """
        num_dummies = len(self.Nodes) - 1
        names_dummies = list(map(lambda i: 'dummy_{}'.format(i), range(num_dummies + 1)))
        names_dummies.pop(self.depotIndex)
        if high_cost is None:
            max_cost = max(a.getCost() for a in self.Arcs.values())
            high_cost = max_cost * (num_dummies + 2)
        cost_dummies = [high_cost] * num_dummies
        columns_dummies = [[[i], [1.0]] for i in range(num_dummies)]
        cplex_prob.variables.add(obj=cost_dummies,
                                 lb=[0.0] * num_dummies,
                                 ub=[1.0] * num_dummies,
                                 types=[cplex_prob.variables.type.binary] * num_dummies,
                                 names=names_dummies,
                                 columns=columns_dummies
                                 )
        return cplex_prob

    def addRoutes_CG(self, add_rate=0.5, explore=0):
        """ Add routes by attempting to solve root node of LP relaxation

        args:
        add_rate (float): Parameter controlling how many routes to add; keep adding routes until the
            number of routes that have actually been added divided by the number of attempts is less
            than add_rate
        explore (float): explore (float): Parameter controlling exploration/sampling during route
            generation

        Return:
        None
        """
        # construct problem
        # And then add dummy variables
        cplex_prob = self.getCplexProb()
        initial_size = cplex_prob.variables.get_num()
        assert initial_size > 0, "Please add routes to problem first"
        num_constraints = cplex_prob.linear_constraints.get_num()
        cplex_prob = self.make_feasible(cplex_prob)

        # Suppress output
        cplex_prob.set_results_stream(None)

        total_num_added = 0
        prev_duals = numpy.zeros(num_constraints)
        dual_change = 1
        while dual_change > 1e-1:

            logger.info("Number of variables: {}".format(cplex_prob.variables.get_num()))
            # Solve continuous relaxation
            cplex_prob.set_problem_type(cplex_prob.problem_type.LP)
            cplex_prob.solve()
            duals = cplex_prob.solution.get_dual_values()
            dual_change = numpy.linalg.norm(prev_duals - duals)
            prev_duals = numpy.asarray(duals)
            logger.info("Dual change: {}".format(dual_change))
            # insert a dummy value for depot index
            duals.insert(self.depotIndex, 0.0)
            node_costs = -numpy.array(duals)

            # Add variables
            vf = None
            num_added = 1
            total_tried = 1
            while num_added / total_tried > add_rate:
                r, vf = self.generateRoute(vf, explore, node_costs)
                added, cost, column = self.addRoute(r)
                red_cost = cost + node_costs.dot(column)
                total_tried += 1
                if added:
                    if red_cost < -1e-3:
                        logger.info("Reduced cost: {}".format(red_cost))
                        num_added += 1
                    name = 'r_' + '_'.join(map(lambda i: '{}'.format(i), r))
                    logger.info("Add rate: {}".format(num_added / total_tried))
                    logger.debug("Route {} added".format(name))
                    logger.debug("Column: {}".format(column))
                    column.pop(self.depotIndex)
                    cplex_prob.variables.add(obj=[cost],
                                             lb=[0.0],
                                             ub=[1.0],
                                             types=cplex_prob.variables.type.continuous,
                                             names=[name],
                                             columns=[[list(range(num_constraints)), column]]
                                             )
            # end inner loop
            total_num_added += (num_added - 1)

        # Delete the dummy vars that were added
        logger.info("Total number of variables added: {}".format(total_num_added))
        return

    def getBLECdata(self):
        """
        Transform problem data into the cost vector, constraint matrix and rhs
        of the Binary Linear Equality Constrained (BLEC) problem
        """
        num_nodes = len(self.Nodes)
        blec_cost = numpy.array(self.route_costs)
        # constraints are: visit all nodes (EXCEPT depot) exactly once
        # route_node_visited is a list of lists essentially giving transpose of
        # constraint matrix; just get rid of row corresponding to depot
        blec_constraints_rhs = numpy.ones(num_nodes - 1)
        blec_constraints_matrix = numpy.array(self.route_node_visited).transpose()
        mask = numpy.ones(num_nodes, dtype=bool)
        mask[self.depotIndex] = False
        blec_constraints_matrix = blec_constraints_matrix[mask, :]
        return blec_cost, blec_constraints_matrix, blec_constraints_rhs

    def getCplexProb(self):
        """ Get a CPLEX object containing the BLEC/MIP representation

        args:
        None

        Return:
        cplex_prob (cplex.Cplex): A CPLEX object defining the MIP problem
        """
        cplex_prob = cplex.Cplex()

        # Get BLEC/MIP data (but convert to lists for CPLEX)
        blec_cost, blec_constraints_matrix, blec_constraints_rhs = self.getBLECdata()
        # Variables: all binary
        # constraints: all equality
        var_types = [cplex_prob.variables.type.binary] * len(blec_cost)
        con_types = ['E'] * len(blec_constraints_rhs)
        (rows, cols) = numpy.nonzero(blec_constraints_matrix)
        vals = blec_constraints_matrix[(rows, cols)]
        rows = rows.tolist()
        cols = cols.tolist()
        vals = vals.tolist()

        # Variable names: node index sequence
        # Given a route (a list of indices), convert to a single string of those indices
        route_namer = lambda r: 'r_' + '_'.join(map(lambda i: '{}'.format(i), r))
        vnames = list(map(route_namer, self.routes))
        # Constraint names: name after node index
        cnames = list(map(lambda i: 'cNode_{}'.format(i), range(len(self.Nodes))))
        cnames.pop(self.depotIndex)

        # define object
        cplex_prob.objective.set_sense(cplex_prob.objective.sense.minimize)
        cplex_prob.variables.add(obj=blec_cost.tolist(), types=var_types, names=vnames)
        cplex_prob.linear_constraints.add(rhs=blec_constraints_rhs.tolist(), senses=con_types,
                                          names=cnames)
        cplex_prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
        return cplex_prob

    def solveCplexProb(self, filename_sol='cplex.sol'):
        cplex_prob = self.getCplexProb()
        cplex_prob.solve()
        cplex_prob.solution.write(filename_sol)
        return

    def getRoutes(self, solution):
        """ Get a representation of the paths/ vehicle routes in a solution

        args:
        solution: binary vector corresponding to a solution

        Return:
        routes (ndarray): 2-dimensional array, whose rows are routes/node sequences
        """
        # Find contiguous routes through the network
        return numpy.array(self.routes)[numpy.nonzero(solution)]

    def getQUBO(self, penalty_parameter=None, feasibility=False):
        """ Get the Quadratic Unconstrained Binary Optimization problem reformulation of the BLEC

        args:
        penalty_parameter (float): value of penalty parameter to use for reformulation. If None, it
            is determined automatically
        feasibility (bool): Get the feasibility problem (ignore the objective)

        Return:
        Q (ndarray): Square matrix defining QUBO
        c (float): a constant that makes the objective of the QUBO equal to the objective of the
            BLEC
        """

        if feasibility:
            # ignore costs, just capture penalized constraints
            penalty_parameter = 1
        else:
            # do exact penalty; look at L1 norm of cost vector
            sufficient_pp = numpy.sum(numpy.abs(self.route_costs))
            if penalty_parameter is None:
                penalty_parameter = sufficient_pp + 1.0
            if penalty_parameter <= sufficient_pp:
                logger.info(
                    "Penalty parameter might not be big enough...(>{})".format(sufficient_pp))

        # num_nodes = len(self.Nodes)
        num_vars = len(self.route_costs)

        _, blec_constraints_matrix, blec_constraints_rhs = self.getBLECdata()

        # according to scipy.sparse documentation,
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix)
        # Duplicated entries are merely summed to together when converting to an array or other sparse matrix type
        # This is consistent with our aim
        qval = []
        qrow = []
        qcol = []

        if not feasibility:
            # Linear objective terms:
            for i in range(len(self.route_costs)):
                if self.route_costs[i] != 0:
                    qrow.append(i)
                    qcol.append(i)
                    qval.append(self.route_costs[i])

        # Linear Equality constraints as penalty:
        # rho*||Ax - b||^2 = rho*( x^T (A^T A) x - 2b^T A x + b^T b )

        # Put -2b^T A on the diagonal:
        TwoBTA = -2 * blec_constraints_matrix.transpose().dot(blec_constraints_rhs)
        for i in range(num_vars):
            if TwoBTA[i] != 0:
                qrow.append(i)
                qcol.append(i)
                qval.append(penalty_parameter * TwoBTA[i])

        # Construct the QUBO objective matrix so far
        # Convert to ndarray, because if we add a dense matrix to it,
        # the result is a numpy.matrix, which is potentially deprecated
        Q = sparse.coo_matrix((qval, (qrow, qcol))).toarray()

        # Add A^T A to it
        # This will be a ndarray
        Q = Q + penalty_parameter * blec_constraints_matrix.transpose().dot(blec_constraints_matrix)
        # constant term of QUBO objective
        constant = penalty_parameter * blec_constraints_rhs.dot(blec_constraints_rhs)
        return Q, constant

    def export_mip(self, filename=None):
        """ Export BLEC/MIP representation of problem """
        if filename is None:
            filename = 'path_based_rp.lp'
        cplex_prob = self.getCplexProb()
        cplex_prob.write(filename)
        return

    def exportQUBO(self, filename=None, feasibility=False):
        """
        DEPRECATED -- use procedure in QUBOTools
        Export the routing problem as a QUBO
        """
        if feasibility:
            # wipe out the costs of the original problem, just capture penalized constraints
            penaltyFactor = 1
            costFactor = 0
        else:
            # do exact penalty
            penaltyFactor = numpy.sum(numpy.abs(self.route_costs)) + 1
            costFactor = 1

        N = len(self.route_costs)
        objectiveConstant = penaltyFactor * (len(self.Nodes) - 1)

        contents = []
        contents.append('c generated ' + str(datetime.datetime.today()))
        contents.append('\nc Constant term of objective = {:.2f}'.format(objectiveConstant))
        contents.append('\nc Program line sentinel follows; format:')
        contents.append('\nc p qubo 0 maxDiagonals nDiagonals nElements')
        # Don't have these data counts yet. Insert later
        SentinelLineIndex = len(contents)
        maxDiagonals = N
        contents.append('\nc Diagonal terms')
        nDiagonals = 0
        for i in range(N):
            # Subtract 1 because we know every route visits the depot and we don't care
            sumdelta = numpy.sum(self.route_node_visited[i]) - 1
            value = costFactor * self.route_costs[i] - penaltyFactor * sumdelta
            if value != 0:
                contents.append('\n{:d} {:d} {: .2f}'.format(i, i, value))
                nDiagonals += 1
        contents.append('\nc (strictly) Upper triangular terms')
        nElements = 0
        for i in range(N):
            for j in range(i + 1, N):
                # Subtract 1 because we know every route visits the depot and we don't care
                sumdeltadelta = numpy.dot(self.route_node_visited[i],
                                          self.route_node_visited[j]) - 1
                value = 2 * penaltyFactor * sumdeltadelta
                if value != 0:
                    contents.append('\n{:d} {:d} {: .2f}'.format(i, j, value))
                    nElements += 1

        # Add in program sentinel
        sentinelLine = '\np qubo 0 {:d} {:d} {:d}'.format(maxDiagonals, nDiagonals, nElements)
        contents.insert(SentinelLineIndex, sentinelLine)
        # Write to file
        if filename is None:
            filename = 'RoutingProblem.qubo'
        file = open(filename, 'w')
        file.write("".join(contents))
        file.close()


def getSampledKey(KeyVal, explore):
    """ Sample the keys of a dictionary inversely proportional to the real values

    The idea is to transform the values into a probability distribution with probability inversely
    proportional to the values, so that the mode corresponds to the smallest value. To do this, use
    a softmax on the negative of the values

    args:
    explore (float): Parameter controlling how much we explore
        explore = 0      : samples more tightly around mode      (makes MORE sensitive to scale)
        explore = np.inf : gives more randomization/exploration  (makes LESS sensitive to scale)

    return:
    sampledKey: A (potentially randomly) sampled key of the dictionary
    minimumKey: The key corresponding to the actual minimum value
    """
    assert KeyVal, 'Dictionary to sample is empty'
    assert explore >= 0, 'explore must be non-negative'

    # Get and return true minimum anyway
    minimumKey = min(KeyVal, key=KeyVal.get)

    # Sample keys according to probabilities obtained from softmax,
    # using appropriate scaling
    scaled_vals = -numpy.fromiter(KeyVal.values(), dtype=float) / (1e-4 + explore)
    pmf = softmax(scaled_vals)
    try:
        sampledKey = numpy.random.choice(list(KeyVal.keys()), p=pmf)
    except(ValueError):
        # make sure pmf is really a pmf
        pmf /= numpy.sum(pmf)
        sampledKey = numpy.random.choice(list(KeyVal.keys()), p=pmf)
    return sampledKey, minimumKey


class Node:
    """
    A node is a customer, with a demand level that must be satisfied in a particular window of time.
    Here, the sign convention for demand is from the perspective of the node:
        positive demand must be delivered,
        negative demand is supply that must be picked up.
    """

    def __init__(self, Name, Demand, TW):
        assert TW[0] <= TW[1], 'Time window for {} not valid: {} > {}'.format(name, TW[0], TW[1])
        self.name = Name
        self.demand = Demand
        self.tw = TW

    def getName(self):
        return self.name

    def getDemand(self):
        """ Return the demand level of the node """
        return self.demand

    def getLoad(self):
        """
        Return what is loaded onto/off a vehicle servicing this node
        (i.e., load is negative demand)
        """
        return -self.demand

    def getWindow(self):
        """ Return the time window (a, b) of the node """
        return self.tw

    def __str__(self):
        return "{}: {} in {}".format(self.name, self.demand, self.tw)


class Arc:
    """
    An arc goes from one node to another (distinct) node
    It has an associated travel time, and potentially a cost
    """

    def __init__(self, From, To, TravelTime, Cost):
        assert From is not To, 'Arc endpoints must be distinct'
        self.origin = From
        self.destination = To
        self.traveltime = TravelTime
        self.cost = Cost

    def getO(self):
        return self.origin

    def getD(self):
        return self.destination

    def getTravelTime(self):
        return self.traveltime

    def getCost(self):
        return self.cost

    def __str__(self):
        return "{} to {}, t={:.2f}".format(self.origin.name, self.destination.name, self.traveltime)
