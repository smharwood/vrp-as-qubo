# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 09:42:01 2018

@author: stuart.m.harwood@exxonmobil.com

Defining a routing problem
Specifically, as seen as a set partitioning problem.
See Desrochers, Desrosiers, Solomon, "A new optimization algorithm for the vehicle routing problem with time windows"

The ultimate goal is to express an instance as a Quadratic Unconstrained Binary Optimization problem
"""

import numpy
import scipy as sp
import datetime

class RoutingProblem:
    """
    The Routing Problem, as a (binary) integer program (IP),
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
        self.NameList = []
        self.Nodes = []
        self.Arcs = dict()
        self.depotIndex = 0 # default depot is zeroth node
        self.vehicleCap = 0
        # Parameters of the MIP representation
        # (although its really an integer program)
        self.mip_costs = []
        self.mip_constraints_T = []
        self.mip_variables = []

    def getNodeIndex(self, NodeName):
        return self.NameList.index(NodeName)
    def getNode(self, NodeName):
        return self.Nodes[self.getNodeIndex(NodeName)]
    def setDepot(self, DepotName):
        self.depotIndex = self.NameList.index(DepotName)
    def setVehicleCap(self, VehicleCap):
        self.vehicleCap = VehicleCap

    def addNode(self,NodeName,Demand,TW=None):
        assert NodeName not in self.NameList, NodeName + ' is already in Node List'
        self.Nodes.append(Node(NodeName,Demand,TW))
        self.NameList.append(NodeName)
    def addArc(self,OName,DName,Time,Cost=0):
        i = self.getNodeIndex(OName)
        j = self.getNodeIndex(DName)
        self.Arcs[(i,j)] = Arc(self.Nodes[i],self.Nodes[j],Time,Cost)

    def checkRoute(self,CandidateRoute):
        """
        Check to see if this path is an allowed route;
        Does it use valid arcs, visit nodes within allowed time windows, and satisfy cumulative demand
        """
        # Convert from list of string node names to indices if necessary
        RouteIndices = CandidateRoute
        if isinstance(CandidateRoute[0], str):
            for i in range(len(CandidateRoute)):
                RouteIndices[i] = self.getNodeIndex(CandidateRoute[i])

        feasible = True
        cost = 0
        visitsNode = [0]*(len(self.Nodes))
        # First check:
        # are first and last nodes the depot?
        if RouteIndices[0] != self.depotIndex or RouteIndices[-1] != self.depotIndex:
            feasible = False
            return feasible, cost, visitsNode
        # Make sure these are valid arcs,
        # visit each node at most once (besides depot)
        # satisfy capacity constraints
        # and time window constraints
        cumuTime = 0
        cumuDemand = 0
        for i in range(len(RouteIndices)-1):
            a = (RouteIndices[i],RouteIndices[i+1])

            # have we already visited this node?
            if visitsNode[RouteIndices[i]] == 1:
                feasible = False
                return feasible, cost, visitsNode
            else :
                visitsNode[RouteIndices[i]] = 1

            # valid arc?
            # will increment cumuTime and cumuDemand appropriately
            feasArc,cumuTime,cumuDemand = self.checkArc(cumuTime,cumuDemand,a)
            if feasArc:
                cost = cost + self.Arcs[a].getCost()
            else:
                feasible = False
                return feasible, cost, visitsNode
        # end for loop
        return feasible, cost, visitsNode

    def checkArc(self,time,demand,arcKey):
        """
        For a given time and demand (and node implied by the arc),
        is the given arc valid/ lead to a valid route?
        """
        feasibleArc = True
        # is the key even valid?
        try :
            arc = self.Arcs[arcKey]
            dest = arc.getD()
        except(KeyError) :
            feasibleArc = False
            return feasibleArc, time, demand
        # Do we arrive before or during the next node's time window?
        # add travel time of current arc,
        # but if we arrive at a node before its time window, we have to wait
        arrivetime = time + arc.getTravelTime()
        arrivetime = max(arrivetime, dest.getWindow()[0])
        if arrivetime > dest.getWindow()[1] :
            feasibleArc = False
        # Does cumulative demand exceed our capacity?
        nextdemand = demand + dest.getDemand()
        if nextdemand > self.vehicleCap:
            feasibleArc = False
        return feasibleArc, arrivetime, nextdemand

    def getRouteNames(self,route):
        """
        given a sequence of indices, print the corresponding node names
        """
        RouteString = []*len(route)
        for index in route:
            RouteString.append(self.NameList[index])
        return RouteString
   
    def generateRoute(self, vf=None, extent=1, noCost=False):
        """
        Generate a route
        Can view this as one iteration of an approximate dynamic programming method
        The goal is to find a good candidate route, and we always have exploration,
            so it is not "true" DP to find an optimal route
        input
            vf: value function over nodes
            extent: controls exploration/sampling
            noCost: do not use arc costs/ stage costs in determining cost-to-go
        """

        if vf is None:
            vf = [0]*len(self.Nodes)
        assert len(vf) == len(self.Nodes), 'Value function incorrect size'
            
        r = []
        # all routes start at depot
        currNode = self.depotIndex
        maxLegs = 2 + len(self.Nodes)
        time = 0
        demand = 0
        # Build up a route
        for r_iter in range(maxLegs):
            r.append(currNode)

            PotentialNodesAndVals = dict()
            # Loop over nodes (really, outgoing arcs),
            # get stage cost plus value function at each node
            for n in range(len(self.Nodes)):
                a = (currNode,n)
                # valid arc?
                feasArc,dumt,dumd = self.checkArc(time,demand,a)
                if feasArc:
                    stageCost = 0 if noCost else self.Arcs[a].getCost()
                    PotentialNodesAndVals[n] = stageCost + vf[n]
                else:
                    continue
            # end for

            # Get a node to go to;
            # Minimize value function
            # Maybe add some randomization, proportional to this objective
            try:
                sampledNode, minNode = getSampledKey(PotentialNodesAndVals,extent)
                a = (currNode,sampledNode)
                feasArc,time,demand = self.checkArc(time,demand,a)
                # Update value function estimate:
                # ACTUAL minimizing cost-to-go: stageCost(a) + vf[minNode]
                vf[currNode] = PotentialNodesAndVals[minNode]                
                currNode = sampledNode
            except AssertionError:
                # this probably shouldn't happen
                break
            # if we have returned to the depot, we are done
            if currNode == self.depotIndex:
                r.append(currNode)
                break
        # end loop over building up route
        return r,vf

    def addRoute(self,route):
        """
        If route is feasible, save its data (cost and constraint column)
        in the MIP representation of the problem
        Returns whether it was added, and the total number of routes visiting
            each node
        """
        # If route r is feasible:
        #  cost = c_r
        #  visitsNode[k] = \delta_k,r
        feas, cost, visitsNode = self.checkRoute(route)
        added = False
        if feas and route not in self.mip_variables:
            self.mip_costs.append(cost)
            self.mip_constraints_T.append(visitsNode)
            self.mip_variables.append(list(route))
            added = True
        NumRoutesVisiting = numpy.sum(self.mip_constraints_T,axis=0)
        return added, NumRoutesVisiting

    def addRoutes(self,maxNumRoutes,explore=1,vf=None):
        """
        Generate and add a batch of routes to problem
        """
        # Value function
        if vf is None:
            vf = [0]*len(self.Nodes)
        assert len(vf) == len(self.Nodes), 'Value function incorrect size'

        # Dynamic programming-like thing
        # Generate a route at each iteration
        # There is always exploration, so not "true" DP
        # but we should be tending to better and better routes
        for i in range(maxNumRoutes):
            r,vf = self.generateRoute(vf,explore)
            added, NumRoutesVisiting = self.addRoute(r)
            NodesVisited = NumRoutesVisiting.tolist()
            # vf is updated in place by generateRoute
        
        if all(NodesVisited):
            print('All nodes covered!')
        else:
            print('Some node not covered by a route')
            
        return vf
            
    def addRoutesFeasibility(self,maxNumRoutes,explore=1):
        """
        Generate and add a batch of routes to problem
        focusing on getting enough different routes to guarantee feasibility
        """
        """
        The value function here measures how many routes have already visited a node
        By minimizing this value function, we generate routes that visit less-visited nodes,
            and so tend to "spread out" over the nodes
        """
        # Value function
        vf = [0]*len(self.Nodes)
        
        # Dynamic programming-like thing
        for i in range(maxNumRoutes):
            r,_ = self.generateRoute(vf,explore,noCost=True)
            added, NumRoutesVisiting = self.addRoute(r)
            NodesVisited = NumRoutesVisiting.tolist()
            # manually write value function
            vf = NumRoutesVisiting.tolist()
            if added:
                print(vf) # to see progression of value function
        
        if all(NodesVisited):
            print('All nodes covered!')
        else:
            print('Some node not covered by a route')

    def exportQUBO(self,filename=None,feasibility=False):
        """
        Export the routing problem, once candidate routes have been added
        Specifically, this will save a file with a sparse matrix
        encoding the Quadratic Binary Unconstrained Optimization (QUBO) problem
        corresponding to the routing problem
        See also
            https://github.com/dwavesystems/qbsolv
        for format that DWave expects, at least
        
        feasibility = True: export just the FEASIBILITY problem
            (essentially, set cost vector to zero)
        """
        """
        The MIP may be rewritten by penalizing the constraints:
        \min_x B(\sum_r c_r x_r) + A(\sum_k (1 - \sum_r \delta_k,r x_r)^2)
          s.t. x_r \in {0,1} for all r

        With some algebra, the penalty is
        A * (number nodes minus depot)
         + \sum_r     x_r     (B c_r + A \sum_i \delta_{i,r}^2 - 2 A \sum_i \delta_{i,r}) + 
         + \sum_{r,s} x_r x_s \sum_i \delta_{i,r} \delta_{i,s}
        
        This is in effect a QUBO:
        \min_x <x, Mx>
          s.t. x_r \in {0,1} for all r,
        noting that 
            x_r = x_r^2 (the variables are {0,1}-valued)
                so linear terms can be added to the diagonal of M,
            and QUBO-as-Ising assumes an UPPER TRIANGULAR matrix,
                so modify M appropriately
        """
        """
        The scaling factors of the cost and penalty terms is important to make it an EXACT penalty
        If we have a feasible solution, the most we can decrease the cost term is (imagine flipping each variable)
         -B \sum_r \abs(c_r)
        Since the elements of \delta_k,r are binary, the smallest a single constraint can be violated is by 1;
        the overall change in QUBO objective is
          A - B \sum_r \abs(c_r)
        To ensure an exact penalty, we need the net change to be positive:
          A - B \sum_r \abs(c_r) > 0
          (it is not favorable to violate constraints)
        Take B = 1, A > \sum_r \abs(c_r)
        """
        if feasibility:
            # wipe out the costs of the original problem, just capture penalized constraints
            penaltyFactor = 1
            costFactor = 0
        else:
            # do exact penalty as above
            penaltyFactor = numpy.sum(numpy.abs(self.mip_costs)) + 1
            costFactor = 1

        N = len(self.mip_variables)
        objectiveConstant = penaltyFactor*(len(self.Nodes)-1)

        contents = []
        contents.append('c generated '+str(datetime.datetime.today()))
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
            sumdelta = numpy.sum(self.mip_constraints_T[i]) - 1
            #sumdeltasquare = numpy.sum(numpy.square(self.mip_constraints_T[i])) - 1
            #assert (sumdelta == sumdeltasquare), "delta should be {0,1}-valued?"
            #value = costFactor*self.mip_costs[i] + penaltyFactor*(sumdeltasquare - 2*sumdelta)
            value = costFactor*self.mip_costs[i] - penaltyFactor*sumdelta
            if value != 0 :
                contents.append('\n{:d} {:d} {: .2f}'.format(i,i,value))
                nDiagonals += 1
        contents.append('\nc (strictly) Upper triangular terms')
        nElements = 0
        for i in range(N):
            for j in range(i+1,N):
                # Subtract 1 because we know every route visits the depot and we don't care
                sumdeltadelta = numpy.dot(self.mip_constraints_T[i],self.mip_constraints_T[j]) - 1
                value = 2*penaltyFactor*sumdeltadelta
                if value != 0:
                    contents.append('\n{:d} {:d} {: .2f}'.format(i,j,value))
                    nElements += 1
        
        # Add in program sentinel
        sentinelLine = '\np qubo 0 {:d} {:d} {:d}'.format(maxDiagonals,nDiagonals,nElements)
        contents.insert(SentinelLineIndex,sentinelLine)
        # Write to file
        if filename is None:
            filename = 'RoutingProblem.qubo'
        file = open(filename,'w')
        file.write("".join(contents))
        file.close()

    def exportIsing(self,filename=None,feasibility=False):
        """
        Export the routing problem, once candidate routes have been added
        Specifically, this will save a file with a sparse matrix
        encoding the ISING model form corresponding to the routing problem
        
        feasibility = True: export just the FEASIBILITY problem
            (essentially, set cost vector to zero)            
        """
        """
        This is achieved by writing
        s_i = 2 x_i - 1
        <==>
        x_i = 0.5(s_i + 1)
        and working through the algebra to transform the penalty function
        Thus the same penalty/cost factors are used
        """
        N = len(self.mip_variables)
        
        if feasibility:
            # wipe out the costs of the original problem, just capture penalized constraints
            penaltyFactor = 1
            costFactor = 0
        else:
            # do exact penalty as above
            penaltyFactor = numpy.sum(numpy.abs(self.mip_costs)) + 1
            costFactor = 1
            
        # Make array copy of self.mip_constraints_T
        # and slice out depot info
        deltas_T_nd= numpy.array(self.mip_constraints_T)
        nodes_keep = list(range(len(self.Nodes)))
        nodes_keep.pop(self.depotIndex)
        deltas_T_nd = deltas_T_nd[:,nodes_keep]
        # Define some variables that sum over the routes
        sum_r_delta = numpy.sum(deltas_T_nd, axis=0)
        gamma = 1 - 0.5*sum_r_delta
        objectiveConstant = 0.5*costFactor*numpy.sum(self.mip_costs) + \
                            penaltyFactor*numpy.sum(numpy.square(gamma)) + \
                            0.25*penaltyFactor*numpy.sum(sum_r_delta)

        contents = []
        contents.append('# generated '+str(datetime.datetime.today()))
        contents.append('\n# Constant term of objective = {:.2f}'.format(objectiveConstant))
        # Don't have number of edges yet. Insert later
        InfoLineIndex = len(contents)
        contents.append('\n# Diagonals / \"h\" vector')
        nEdges = 0
        for r in range(N):
            value = costFactor*0.5*self.mip_costs[r] - penaltyFactor*numpy.dot(gamma,deltas_T_nd[r])
            if value != 0 :
                contents.append('\n{:d} {:d} {: .2f}'.format(r,r,value))
                nEdges += 1
        contents.append('\n# (strictly) Upper triangular terms / \"J\" matrix')
        for r in range(N):
            for rp in range(r+1,N):
                sumdeltadelta = numpy.dot(deltas_T_nd[r],deltas_T_nd[rp])
                value = 0.5*penaltyFactor*sumdeltadelta
                if value != 0:
                    contents.append('\n{:d} {:d} {: .2f}'.format(r,rp,value))
                    nEdges += 1
        
        # Add in info line: number of spins(variables) and edges
        infoLine = '\n# nVariables nEdges: \n{:d} {:d}'.format(N,nEdges)
        contents.insert(InfoLineIndex,infoLine)
        # Write to file
        if filename is None:
            filename = 'RoutingProblem.rudy'
        file = open(filename,'w')
        file.write("".join(contents))
        file.close()
        
    def exportMPS(self,filename=None):
        """
        Export MIP representation as MPS file for testing
        """
        N = len(self.mip_variables)
        M = len(self.Nodes)
        if filename is None:
            filename = 'RoutingProblem.mps'
        file = open(filename,'w')
        file.write('{:<14}{}'.format('NAME',filename))
        file.write('\nROWS')
        file.write('\n N  obj')
        for k in range(M):
            if k == self.depotIndex:
                continue
            file.write('\n E  n{}'.format(k))
        file.write('\nCOLUMNS')
        file.write('\n{:4}{:<10}{:<10}{:15}{:<}'.format('','MARK0','\'MARKER\'','','\'INTORG\''))
        for i in range(N):
            file.write('\n{:4}x{:<9}{:<10}{:<}'.format('',i,'obj',self.mip_costs[i]))
            for k in range(M):
                if k == self.depotIndex:
                    continue
                value = self.mip_constraints_T[i][k]
                if value == 0:
                    continue
                file.write('\n{:4}x{:<9}n{:<9}{:<}'.format('',i,k,value))
        file.write('\n{:4}{:<10}{:<10}{:15}{:<}'.format('','MARK1','\'MARKER\'','','\'INTEND\''))
        file.write('\nRHS')
        for k in range(M):
            if k == self.depotIndex:
                continue
            file.write('\n{:4}{:<10}n{:<9}{:<}'.format('','rhs',k,1))
        # Default bounds for integer variables should be [0,1]-
        # thus enforcing a binary variable
        # Can enforce binary with bounds section too:
#        file.write('\nBOUNDS')
#        for i in range(N):
#            file.write('\n BV {:<10}x{:<9}'.format('BOUND',i))
        file.write('\nENDATA')
        file.close()


def getSampledKey(KeyVal,extent):
    """
    Simple routine;
    given a dictionary with keys and real values,
    sample the keys inversely proportional to the values
    """
    """
    The idea is to transform the values into a probability distribution 
    with probability inversely proportional to the values,
    so that the mode corresponds to the smallest value.
    To do this, use a softmax on the negative of the values
    """
    assert KeyVal, 'Dictionary to sample is empty'
    assert extent >= 0, 'extent must be non-negative'
    
    # Get and return true minimum anyway
    minimumKey = min(KeyVal, key=KeyVal.get)

    # 'extent' is Extent to which we want to scale the values
    # extent = 0      samples more tightly around mode      (makes MORE sensitive to scale)
    # extent -> infty gives more randomization/exploration  (makes LESS sensitive to scale)
    scaling = \
            max(abs(min(KeyVal.values())),abs(max(KeyVal.values())))
            #max(numpy.abs(list(KeyVal.values())))
    # squash thru softmax (and take negative)
    # to get probability mass function
    pmf = { k:numpy.exp(-KeyVal[k]/(1 + extent*scaling)) for k in KeyVal.keys() }
    total = sum(pmf.values())
    # generate uniform random number
    ur = total*numpy.random.uniform()
    runTotal = 0
    # Use inverse transform sampling of pmf
    for (sampleKey,val) in pmf.items():
        runTotal += val
        if runTotal >= ur:
            return sampleKey, minimumKey


class Node:
    """
    A node is a customer,
        with an amount of product demanded/that must be delivered,
        and in a particular window of time
    The exception is the "depot" node,
        from which vehicles start and finish their route.
        It has no demand or meaningful time window
    """
    def __init__(self,Name,Demand,TW=None):
        self.name = Name
        self.demand = Demand
        if TW is None:
            self.tw = (0,numpy.inf)
        else:
            assert TW[0] < TW[1], 'Time window for '+Name+' not valid'
            self.tw = TW
    def getName(self):
        return self.name
    def getDemand(self):
        return self.demand
    def getWindow(self):
        return self.tw
    def __str__(self):
        return "{}: {} in {}".format(self.name,self.demand,self.tw)

class Arc:
    """
    An arc goes from one node to another (distinct) node
    It has an associated travel time, and potentially a cost
    """
    def __init__(self,From,To,TravelTime,Cost):
        assert From is not To, 'Arc endpoints must be distinct'
        self.origin = From
        self.destination = To
        self.traveltime = TravelTime
        self.cost = Cost
        # if Cost is None:
            # # default: define cost as travel time
            # self.cost = self.traveltime
        # else:
            # self.cost = Cost
    def getO(self):
        return self.origin
    def getD(self):
        return self.destination
    def getTravelTime(self):
        return self.traveltime
    def getCost(self):
        return self.cost