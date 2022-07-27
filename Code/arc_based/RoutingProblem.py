# -*- coding: utf-8 -*-
"""
Created on 19 December 2019

@author: stuart.m.harwood@exxonmobil.com

Container for a arc-based formulation of a 
Vehicle Routing Problem with Time Windows
"""
import time
import numpy
import scipy.sparse as sparse
try:
    import cplex
except ImportError:
    pass

class RoutingProblem:
    """
    Vehicle Routing Problem with Time Windows (VRPTW)
    as a discrete-time arc-based formulation
    is a Binary Linear Equality-Constrained  (BLEC)
    optimization problem
    which can be transformed to a
    Quadratic Unconstrained Binary Optimization (QUBO) problem
    """

    def __init__(self):
        # basic data
        self.Nodes = []
        self.NodeNames = []
        self.Arcs = dict()
        self.TimePoints = []
        self.VarMapping = []
        self.variablesEnumerated = False
        self.constraints_built = False
        self.objective_built = False

        # Parameters of the base formulation:
        # Binary Linear Equality-Constrained (BLEC) problem
        self.blec_c = None
        self.blec_constraints_matrix = None
        self.blec_constraints_rhs = None

    def addNode(self,NodeName,TW):
        assert NodeName not in self.NodeNames, NodeName + ' is already in Node List'
        self.Nodes.append(Node(NodeName,TW))
        self.NodeNames.append(NodeName)

    def addDepot(self,DepotName,TW):
        """Insert depot at first position of nodes"""
        if not numpy.isinf(TW[1]):
            print("Consider making Depot time window infinite in size...")
        try:
            if self.NodeNames[0] != DepotName:
                self.Nodes.insert(0,Node(DepotName,TW))
                self.NodeNames.insert(0,DepotName)
            else:
                print("Depot already added")
        except(IndexError):
            self.addNode(DepotName, TW)

    def getNodeIndex(self, NodeName):
        return self.NodeNames.index(NodeName)

    def getNode(self, NodeName):
        return self.Nodes[self.getNodeIndex(NodeName)]

    def addArc(self,OName,DName,time,cost=0):
        """
        Add a potentially allowable arc;
        we also check feasibility of TIMING:
            allow if origin TimeWindow start plus travel time < destination TimeWindow end
        """
        i = self.getNodeIndex(OName)
        j = self.getNodeIndex(DName)
        departure_time = self.Nodes[i].getWindow()[0]
        # Add arc if timing works:
        if departure_time + time <= self.Nodes[j].getWindow()[1]:
            self.Arcs[(i,j)] = Arc(self.Nodes[i],self.Nodes[j],time,cost)
        return

    def checkArc(self,arcKey):
        """ Is the arc valid? """
        try :
            self.Arcs[arcKey]
            return True
        except(KeyError) :
            return False

    def checkNodeTimeCompat(self, nodeIndex, timeIndex):
        """ is this node-time pair compatible? """
        TW = self.Nodes[nodeIndex].getWindow()
        return timeIndex >= TW[0] and timeIndex <= TW[1]

    def addTimePoints(self, timepoints):
        """ Populate the valid time points """  
        # Copy the SORTED timepoints; this can be a sequence, a numpy array, whatever.
        # Variables are indexed by a sequence which can be made of anything,
        # but we do assume they are sorted for some convenience
        self.TimePoints = numpy.sort(timepoints)
        return


    def enumerateVariables(self):
        if self.variablesEnumerated:
            return
        start = time.time()
        self.enumerateVariables_quicker()
        #self.enumerateVariables_exhaustive()
        duration = time.time() - start
        print("Variable enumeration took {} seconds".format(duration))
        return

    def enumerateVariables_quicker(self):
        """ Basic operation that needs to be done to keep track of variable counts, indexing """
        num_vars = 0
        # Loop over (i,s,j,t)
        # and check if a variable is allowed (nonzero)
        # Simplification: loop over arcs
        for (i,j) in self.Arcs.keys():
            for s in self.TimePoints:
                # First check:
                # is s in the time window of i?
                if s < self.Nodes[i].getWindow()[0]:
                    continue
                if s > self.Nodes[i].getWindow()[1]:
                    # TimePoints is sorted, so s will keep increasing in this loop
                    # This condition will continue to be satisfied
                    break
                for t in self.TimePoints:
                    # Second check:
                    # is t in the time window of j?
                    if t < self.Nodes[j].getWindow()[0]:
                        continue
                    if t > self.Nodes[j].getWindow()[1]:
                        # Same as with s loop: t will keep increasing in this loop
                        break
                    # Third check
                    # is the travel time from i to j consistent with the timing?
                    if s + self.Arcs[(i,j)].getTravelTime() > t:
                        continue

                    # at this point, the tuple (i,s,j,t) is allowed
                    # record the index
                    self.VarMapping.append((i,s,j,t))
                    num_vars += 1
        # end loops
        self.NumVariables = num_vars
        self.variablesEnumerated = True
        return

    def enumerateVariables_exhaustive(self):
        """ Basic operation that needs to be done to keep track of variable counts, indexing """
        num_vars = 0
        # Loop over (i,s,j,t)
        # and check if a variable is allowed (nonzero)
        for i in range(len(self.Nodes)):
            for s in self.TimePoints:
                # first check:
                # is s in the time window of i?
                #if s < self.Nodes[i].getWindow()[0] or s > self.Nodes[i].getWindow()[1]:
                if not self.checkNodeTimeCompat(i,s):
                    continue
                for j in range(len(self.Nodes)):
                    # second check:
                    # is (i,j) an allowed arc?
                    if not self.checkArc((i,j)):
                        continue
                    for t in self.TimePoints:
                        # third check:
                        # is t in the time window of j?
                        #if t < self.Nodes[j].getWindow()[0] or t > self.Nodes[j].getWindow()[1]:
                        if not self.checkNodeTimeCompat(j,t):
                            continue
                        # fourth check
                        # is the travel time from i to j consistent with the timing?
                        if s + self.Arcs[(i,j)].getTravelTime() > t:
                            continue

                        # at this point, the tuple (i,s,j,t) is allowed
                        # record the index
                        self.VarMapping.append((i,s,j,t))
                        num_vars += 1
        # end loops
        self.NumVariables = num_vars
        self.variablesEnumerated = True
        return

    def getNumVariables(self):
        """ number of variables in formulation """
        return self.NumVariables

    def getVarIndex(self, ONodeIndex, OTimeIndex, DNodeIndex, DTimeIndex):
        """Get the unique id/index of the binary variable given the "tuple" indexing
           Return of None means the tuple does not correspond to a variable """
        try:
            return self.VarMapping.index((ONodeIndex, OTimeIndex, DNodeIndex, DTimeIndex))
        except(ValueError):
            return None

    def getVarTupleIndex(self, vIndex):
        """Inverse of getVarIndex"""
        try:
            return self.VarMapping[vIndex]
        except(IndexError):
            return None

    def build_blec_obj(self): 
        """
        Build up linear objective of base BLEC formulation
        """
        if self.objective_built:
           # Objective already built
            return

        self.enumerateVariables()
        # linear terms 
        self.blec_c = numpy.zeros(self.getNumVariables())
        for k in range(self.getNumVariables()):
            (i,s,j,t) = self.VarMapping[k]
            self.blec_c[k] = self.Arcs[(i,j)].getCost()
        self.objective_built = True
        return

    def build_blec_constraints(self):
        if self.constraints_built:
            # Constraints already built
            return
        start = time.time()
        self.build_blec_constraints_quicker()
        #self.build_blec_constraints_exhaustive()
        duration = time.time() - start
        print("Constraint construction took {} seconds".format(duration))
        return

    def build_blec_constraints_exhaustive(self):
        """
        Build up Linear equality constraints of BLEC
        A*x = b

        SLOW but probably correct? Might add a bunch of vacuous constraints
        """
        self.enumerateVariables()

        aval = []
        arow = []
        acol = []
        brhs = []
        row_index = 0
        
        # Flow conservation constraints (for each (i,s))
        # EXCEPT DEPOT
        # see above- Depot is first in node list
        for i in range(1,len(self.Nodes)):
            for s in self.TimePoints:
                # sum_jt x_jtis - sum_jt x_isjt = 0
                for j in range(len(self.Nodes)):
                    for t in self.TimePoints:   
                        col = self.getVarIndex(j,t,i,s)
                        if col is not None:
                            aval.append(1)
                            arow.append(row_index)
                            acol.append(col)
                        col = self.getVarIndex(i,s,j,t)
                        if col is not None:
                            aval.append(-1)
                            arow.append(row_index)
                            acol.append(col)
                # end construction of row entries
                brhs.append(0)
                row_index += 1

        # Sevicing/visitation constraints (for each j)
        # EXCEPT DEPOT (again, depot is first in node list)
        for j in range(1,len(self.Nodes)):
            # sum_ist x_isjt = 1
            for i in range(len(self.Nodes)):
                for s in self.TimePoints:
                    for t in self.TimePoints:   
                        col = self.getVarIndex(i,s,j,t)
                        if col is not None:
                            aval.append(1)
                            arow.append(row_index)
                            acol.append(col)
            # end construction of row entries
            brhs.append(1)
            row_index += 1  

        self.blec_constraints_matrix = sparse.coo_matrix((aval,(arow,acol)))
        self.blec_constraints_rhs = numpy.array(brhs)
        self.constraints_built = True
        return

    def build_blec_constraints_quicker(self):
        """
        Build up Linear equality constraints of BLEC
        A*x = b

        FASTER
        """
        self.enumerateVariables()

        aval = []
        arow = []
        acol = []
        brhs = []
        row_index = 0
        self.constraint_names = []
        
        # Flow conservation constraints (for each (i,s))
        # EXCEPT DEPOT
        # see above- Depot is first in node list
        # First, index the non-trivial constraints
        flow_conservation_mapping = []
        for i in range(1,len(self.Nodes)):
            for s in self.TimePoints:
                # Constraint:
                # sum_jt x_jtis - sum_jt x_isjt = 0

                # is s in the time window of i?
                # (if not, this is a vacuous constraint)
                if s < self.Nodes[i].getWindow()[0]:
                    continue
                if s > self.Nodes[i].getWindow()[1]:
                    # TimePoints is sorted, so s will keep increasing in this loop
                    # This condition will continue to be satisfied
                    # (and so s NOT in time window)
                    break
                flow_conservation_mapping.append((i,s))
                brhs.append(0)
                self.constraint_names.append("cflow_{},{}".format(i,s))
                row_index += 1
        # NOW, go through variables
        # Note: each variable is an arc, and participates in (at most) TWO constraints:
        # once for INflow to a node, and once for OUTflow from a node
        for col in range(self.getNumVariables()):
            (i,s,j,t) = self.getVarTupleIndex(col)
            # OUTflow:
            try:
                row = flow_conservation_mapping.index((i,s))
                aval.append(-1)
                arow.append(row)
                acol.append(col)
            except(ValueError):
                pass
            # INflow:
            try:
                row = flow_conservation_mapping.index((j,t))
                aval.append(1)
                arow.append(row)
                acol.append(col)
            except(ValueError):
                pass

        # Servicing/visitation constraints (for each j)
        # EXCEPT DEPOT (again, depot is first in node list)
        # sum_ist x_isjt = 1
        brhs = brhs + [1]*(len(self.Nodes) - 1)
        for j in range(1,len(self.Nodes)):
            self.constraint_names.append("cnode{}".format(j))
        for col in range(self.getNumVariables()):
            (i,s,j,t) = self.getVarTupleIndex(col)
            # We don't care about how many times depot is visited
            if j == 0:
                continue
            aval.append(1)
            arow.append(row_index + (j-1))
            acol.append(col)

#        # Original version (slower)
#        for j in range(1,len(self.Nodes)):
#            # sum_ist x_isjt = 1
#            for col in range(self.getNumVariables()):
#                (i,s,jp,t) = self.getVarTupleIndex(col)
#                if jp == j:
#                    aval.append(1)
#                    arow.append(row_index)
#                    acol.append(col)
#            # end construction of row entries
#            brhs.append(1)
#            row_index += 1  

        self.blec_constraints_matrix = sparse.coo_matrix((aval,(arow,acol)))
        self.blec_constraints_rhs = numpy.array(brhs)
        self.constraints_built = True
        return

    def getRoutes(self, solution):
        """
        Get a representation of the paths/ vehicle routes in a solution
        
        solution: binary vector corresponding to a solution       
        """
        soln_var_indices = numpy.nonzero(solution)
        soln_var_indices = soln_var_indices[0]
        # Lexicographically sort the indices; all routes start from Depot (node index zero), 
        # so sort the arcs so that those leaving the depot are first
        # Flip the tuples because numpy.lexsort sorts on last row, second to last row, ...
        soln_var_tuples = [self.getVarTupleIndex(k) for k in soln_var_indices]
        tuples_to_sort = numpy.flip(numpy.array(soln_var_tuples), -1)
        arg_sorted = numpy.lexsort(tuples_to_sort.T)
        tuples_ordered = [soln_var_tuples[i] for i in arg_sorted]
        #print(tuples_ordered)
        # While building route, do some dummy checks to make sure formulation is right;
        # check that each node is visited exactly once
        visited = numpy.zeros(len(self.Nodes))
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
                assert self.checkNodeTimeCompat(arc[2], arc[3]), "Node time window not satisfied"
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
        

    def getCplexProb(self):
        """
        Get a CPLEX object containing the BLEC/MIP representation
        """
        self.build_blec_obj()
        self.build_blec_constraints()

        cplex_prob = cplex.Cplex()

        # Variables: all binary
        # constraints: all equality
        var_types = [cplex_prob.variables.type.binary] * len(self.blec_c)
        namer = lambda isjt: "n{}t{}_n{}t{}".format(isjt[0], isjt[1], isjt[2], isjt[3])
        names = list(map(namer, self.VarMapping))
        con_types = ['E'] * len(self.blec_constraints_rhs)
        rows = self.blec_constraints_matrix.row.tolist()
        cols = self.blec_constraints_matrix.col.tolist()
        vals = self.blec_constraints_matrix.data.tolist()

        # define object
        cplex_prob.objective.set_sense(cplex_prob.objective.sense.minimize)
        cplex_prob.variables.add(obj=self.blec_c.tolist(), types=var_types, names=names)
        cplex_prob.linear_constraints.add(rhs=self.blec_constraints_rhs.tolist(), senses=con_types,
            names=self.constraint_names)
        cplex_prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
        return cplex_prob

    def solveCplexProb(self, filename_sol='cplex.sol'):
        cplex_prob = self.getCplexProb()
        cplex_prob.solve()
        cplex_prob.solution.write(filename_sol)
        return

    def getConstraintData(self):
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
        self.build_blec_constraints()
        A_eq = self.blec_constraints_matrix
        b_eq = self.blec_constraints_rhs
        n = self.getNumVariables()
        # if anything is empty, make sure its dense
        if len(b_eq) == 0: A_eq = A_eq.toarray()
        return A_eq, b_eq, sparse.csr_matrix((n,n)), 0

    def getQUBO(self, penalty_parameter=None, feasibility=False):
        """
        Get the Quadratic Unconstrained Binary Optimization problem
        reformulation of the BLEC
        penalty_parameter : value of penalty parameter to use for reformulation
            Default: None (determined by arc costs)
        """
        self.build_blec_obj()
        self.build_blec_constraints()

        if feasibility:
            penalty_parameter = 1.0
        else:
            sum_arc_cost = sum([numpy.fabs(arc.getCost()) for arc in self.Arcs.values()])
            sufficient_pp = (len(self.TimePoints)**2)*sum_arc_cost
            if penalty_parameter is None:
                penalty_parameter = sufficient_pp + 1.0
            if penalty_parameter <= sufficient_pp:
                print("Penalty parameter might not be big enough...(>{})".format(sufficient_pp))

        qval = []
        qrow = []
        qcol = []

        # according to scipy.sparse documentation,
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix)
        # Duplicated entries are merely summed to together when converting to an array or other sparse matrix type
        # This is consistent with our aim
        
        # Linear objective terms:
        if not feasibility:
            for i in range(self.getNumVariables()):
                if self.blec_c[i] != 0:
                    qrow.append(i)
                    qcol.append(i)
                    qval.append(self.blec_c[i])

        # Linear Equality constraints:
        # rho * ||Ax - b||^2 = rho*( x^T (A^T A) x - 2b^T A x + b^T b )

        # Put -2b^T A on the diagonal:
        TwoBTA = -2*self.blec_constraints_matrix.transpose().dot(self.blec_constraints_rhs)     
        for i in range(self.getNumVariables()):
            if TwoBTA[i] != 0:
                qrow.append(i)
                qcol.append(i)
                qval.append(penalty_parameter*TwoBTA[i])

        # Construct the QUBO objective matrix so far
        Q = sparse.coo_matrix((qval,(qrow,qcol)), shape=(self.getNumVariables(),self.getNumVariables()) )

        # Add A^T A to it
        # This will be some sparse matrix (probably CSR format)
        Q = Q + penalty_parameter*self.blec_constraints_matrix.transpose().dot(self.blec_constraints_matrix)

        # constant term of QUBO objective
        constant = penalty_parameter*self.blec_constraints_rhs.dot(self.blec_constraints_rhs)

        return Q, constant

    def export_mip(self, filename=None):
        """ Export BLEC/MIP representation of problem """
        if filename is None:
            filename = 'arc_based_rp.lp'
        cplex_prob = self.getCplexProb()
        cplex_prob.write(filename)
        return


class Node:
    """
    A node is a customer, which must be visited in a particular window of time
    """
    def __init__(self,Name,TW):
        assert TW[0] <= TW[1], 'Time window for '+Name+' not valid'
        self.name = Name
        self.tw = TW
    def getName(self):
        return self.name
    def getWindow(self):
        return self.tw
    def __str__(self):
        return self.name
    def __repr__(self):
        return "{}: in {}".format(self.name, self.tw)


class Arc:
    """
    An arc goes from one node to another (distinct) node
    It has an associated travel time, and a cost
    """
    def __init__(self,From,To,TravelTime,Cost):
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
        return "{} to {}; time={}".format(self.origin.getName(),self.destination.getName(),self.traveltime)

