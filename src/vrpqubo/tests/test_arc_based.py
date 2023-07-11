"""
SM Harwood
19 October 2022
"""
import logging
import unittest
import numpy as np
from ..routing_problem import ArcBasedRoutingProblem

class TestArcBased(unittest.TestCase):
    """ Test elements of arc_based """
    logger = logging.getLogger(__name__)

    def test(self):
        """ Test construction of very simple arc-based problem """
        # Very simple example
        ab = ArcBasedRoutingProblem()
        ab.add_node("depot", 0)
        ab.add_node("node1", 1, (1,2))
        ab.add_node("node2", 1, (3,4))
        ab.add_arc("depot", "node1", 1, 1)
        ab.add_arc("node1", "node2", 1, 1)
        ab.add_arc("node2", "depot", 3, 3)
        self.assertEqual(3, len(ab.nodes), "Number of nodes is incorrect")
        self.assertEqual(3, len(ab.arcs), "Number of arcs is incorrect")

        # Time points should get orderd
        ab.add_time_points([6,3,1,0])
        self.assertEqual(0, np.linalg.norm(ab.time_points - [0,1,3,6]),
            "Time points are incorrect"
        )

        # Variables: arcs between (node,time) pairs
        # (depot,0) - (node1,1)
        # (node1,1) - (node2,3)
        # (node2,3) - (depot,6)
        self.assertEqual(3, ab.get_num_variables(), "Number of variables is incorrect")

        # Cost vector
        depot_index = ab.get_node_index("depot")
        node1_index = ab.get_node_index("node1")
        node2_index = ab.get_node_index("node2")
        dn1_index = ab.get_var_index(depot_index, 0, node1_index, 1)
        n1n2_index = ab.get_var_index(node1_index, 1, node2_index, 3)
        n2d_index = ab.get_var_index(node2_index, 3, depot_index, 6)
        # Individual costs are costs of arcs
        c = np.zeros(3)
        c[dn1_index] = 1
        c[n1n2_index] = 1
        c[n2d_index] = 3
        c_obj, Q_obj = ab.get_objective_data()
        self.assertEqual(0, np.linalg.norm(c_obj - c), "Objective is incorrect")
        self.assertEqual(0, Q_obj.nnz, "Too many quadratic terms in objective")

        # Feasible solution: use each variable
        x = np.ones(3)
        # test whether constraints are satisfied
        A_eq, b_eq, Q_eq, r_eq = ab.get_constraint_data()
        self.assertAlmostEqual(0, np.linalg.norm(A_eq.dot(x) - b_eq),
            "Linear constraints not satisfied"
        )
        # Not expecting any quadratic constraints
        self.assertEqual(0, Q_eq.nnz, "Too many quadratic constraints")
        self.assertEqual(0, r_eq, "Quadratic constraint data is wrong")

        # QUBO matrix: cᵀx + xᵀQx + ρ(||Ax - b||² + xᵀRx)
        pp = 100.0
        Q, c = ab.get_qubo(feasibility=False, penalty_parameter=pp)
        self.assertEqual(c, pp*b_eq.dot(b_eq), "Constant of QUBO is wrong")
        Q_manual = pp*(A_eq.transpose().dot(A_eq)).toarray()
        Q_manual += np.diag(-2*pp*A_eq.transpose().dot(b_eq) + c_obj)
        self.assertEqual(0, np.linalg.norm(Q - Q_manual), "QUBO matrix is incorrect")
