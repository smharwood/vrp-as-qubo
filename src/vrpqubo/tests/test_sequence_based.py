"""
SM Harwood
19 October 2022
"""
import logging
import unittest
import numpy as np
from ..routing_problem import SequenceBasedRoutingProblem

class TestSequenceBased(unittest.TestCase):
    """ Test elements of sequence_based """
    logger = logging.getLogger(__name__)

    def test(self):
        """ Test construction of very simple sequence-based problem """
        # Very simple example
        sb = SequenceBasedRoutingProblem()
        sb.add_node("depot", 0)
        sb.add_node("node1", 1, (1,2))
        sb.add_node("node2", 1, (3,4))
        sb.add_arc("depot", "node1", 1, 1)
        sb.add_arc("node1", "node2", 1, 1)
        sb.add_arc("node2", "depot", 3, 3)
        self.assertEqual(3, len(sb.nodes), "Number of nodes is incorrect")
        self.assertEqual(3, len(sb.arcs), "Number of arcs is incorrect")
        # Setting depot should add (depot,depot) arc
        sb.set_depot("depot")
        self.assertEqual(4, len(sb.arcs), "Number of arcs is incorrect")

        # Variables: degrees of freedom are:
        # (node 1, seq 1)
        # (node 2, seq 2)
        # (depot, seq 1)
        # (depot, seq 2)
        sb.set_max_vehicles(1)
        sb.set_max_sequence_length(4)
        n = sb.get_num_variables()
        self.assertEqual(4, n, "Number of variables is wrong")

        # Cost vector and feasible solution
        depot_index = sb.get_node_index("depot")
        node1_index = sb.get_node_index("node1")
        node2_index = sb.get_node_index("node2")
        x = np.zeros(n)
        c_manual = np.zeros(n)
        Q_manual = np.zeros((n,n))
        # Variable indices: vehicle, sequence, node
        v1_index = sb.get_var_index(0, 1, node1_index)
        self.assertFalse(v1_index is None, "Incorrect variable index")
        c_manual[v1_index] = 1 # cost of depot - node1
        x[v1_index] = 1
        v2_index = sb.get_var_index(0, 2, node2_index)
        self.assertFalse(v2_index is None, "Incorrect variable index")
        c_manual[v2_index] = 3 # cost of node2 - depot
        Q_manual[v1_index, v2_index] = 1 # cost of node1 - node2
        x[v2_index] = 1
        v_index = sb.get_var_index(0, 1, depot_index)
        self.assertFalse(v_index is None, "Incorrect variable index")
        v_index = sb.get_var_index(0, 2, depot_index)
        self.assertFalse(v_index is None, "Incorrect variable index")

        c_obj, Q_obj = sb.get_objective_data()
        self.assertEqual(0, np.linalg.norm(c_obj - c_manual), "Objective is incorrect")
        self.assertEqual(0, np.linalg.norm(Q_obj - Q_manual),
            "Quadratic objective is incorrect?"
        )
        # Q_manual might be transpose of Q_obj...
        diff = min(np.linalg.norm(Q_obj - Q_manual), np.linalg.norm(Q_obj - Q_manual.T))
        self.assertEqual(0, diff, "Not even the transpose is right")

        # test whether constraints are satisfied
        A_eq, b_eq, Q_eq, r_eq = sb.get_constraint_data()
        self.assertAlmostEqual(0, np.linalg.norm(A_eq.dot(x) - b_eq),
            "Linear constraints not satisfied"
        )
        self.assertAlmostEqual(0, Q_eq.dot(x).dot(x) - r_eq,
            "Quadratic constraints not satisfied"
        )

        # QUBO matrix: cᵀx + xᵀQx + ρ(||Ax - b||² + xᵀRx)
        pp = 100.0
        Q, c = sb.get_qubo(feasibility=False, penalty_parameter=pp)
        self.assertEqual(c, pp*b_eq.dot(b_eq), "Constant of QUBO is wrong")
        Q_manual = (Q_obj + pp*(A_eq.transpose().dot(A_eq) + Q_eq)).toarray()
        Q_manual += np.diag(c_obj - pp*2*A_eq.transpose().dot(b_eq))
        self.assertEqual(0, np.linalg.norm(Q - Q_manual), "QUBO matrix is incorrect")
