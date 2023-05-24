"""
SM Harwood
19 October 2022
"""
import logging
import unittest
import numpy as np
from ..routing_problem import PathBasedRoutingProblem
from ..routing_problem.formulations.path_based_rp import get_sampled_key as sampler

class TestPathBased(unittest.TestCase):
    """ Test elements of path_based_rp """
    logger = logging.getLogger(__name__)

    def test(self):
        """ Test construction of very simple path-based problem """
        # Very simple example
        pb = PathBasedRoutingProblem()
        pb.add_node("depot", 0)
        pb.add_node("node1", 1, (1,2))
        pb.add_node("node2", 1, (3,4))
        pb.add_arc("depot", "node1", 1, 1)
        pb.add_arc("node1", "node2", 1, 1)
        pb.add_arc("node2", "depot", 3, 3)
        pb.add_arc("node1", "depot", 1, 1)
        self.assertEqual(3, len(pb.nodes), "Number of nodes is incorrect")
        self.assertEqual(4, len(pb.arcs), "Number of arcs is incorrect")

        # Variables
        pb.set_initial_loading(2)
        pb.set_vehicle_cap(2)
        _, added = pb.add_route(["depot", "node1", "node2", "depot"])
        self.assertTrue(added, "Route not added")
        _, added = pb.add_route(["depot", "node1", "depot"])
        self.assertTrue(added, "Route was added")
        self.assertEqual(2, pb.get_num_variables(), "Number of variables is incorrect")

        # Objective
        c_manual = np.array([1+1+3, 1+1])
        c_obj, Q_obj = pb.get_objective_data()
        self.assertEqual(0, np.linalg.norm(c_obj - c_manual), "Objective is incorrect")
        self.assertEqual(0, Q_obj.nnz, "Too many quadratic terms in objective")

        # Feasible solution: use first route
        # Infeasible solution: use both routes
        x = np.array([1, 0])
        x_infeas = np.array([1,1])
        # test whether constraints are satisfied
        A_eq, b_eq, Q_eq, r_eq = pb.get_constraint_data()
        self.assertAlmostEqual(0, np.linalg.norm(A_eq.dot(x) - b_eq),
            msg="Linear constraints not satisfied"
        )
        self.assertGreater(np.linalg.norm(A_eq.dot(x_infeas) - b_eq), 0,
            "Linear constraints are satisfied, but shouldn't"
        )
        # Not expecting any quadratic constraints
        self.assertEqual(0, Q_eq.nnz, "Too many quadratic constraints")
        self.assertEqual(0, r_eq, "Quadratic constraint data is wrong")

        # QUBO matrix: cᵀx + xᵀQx + ρ(||Ax - b||² + xᵀRx)
        pp = 10.0
        Q, c = pb.get_qubo_new(feasibility=False, penalty_parameter=pp)
        self.assertEqual(c, pp*b_eq.dot(b_eq), "Constant of QUBO is wrong")
        Q_manual = pp*(A_eq.transpose().dot(A_eq)).toarray()
        Q_manual += np.diag(-2*pp*A_eq.transpose().dot(b_eq) + c_obj)
        self.assertEqual(0, np.linalg.norm(Q - Q_manual), "QUBO matrix is incorrect")

    def test_sampler(self):
        """ Test sampler in path_based_rp """
        test_dict = { 'a':100, 'b':101, 'c':102, 'd':10 }
        sample_counts = { k : 0 for k in test_dict.keys() }
        min_counts = { k : 0 for k in test_dict.keys() }
        N = 1000
        for _ in range(N):
            k_samp, k_min = sampler(test_dict, explore=0)
            sample_counts[k_samp] += 1
            min_counts[k_min] += 1
        self.logger.debug("Sample counts: %s", sample_counts)
        self.assertEqual(min_counts['d'], N,
            "Wrong number of counts for minimum"
        )
        # with no exploration, this should be equal
        self.assertEqual(sample_counts, min_counts,
            "Sample counts with no exploration is wrong"
        )
