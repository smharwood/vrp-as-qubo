"""
SM Harwood
19 October 2022
"""
import logging
import unittest
import numpy as np
from ..examples.small import (
    get_arc_based,
    get_path_based,
    get_sequence_based,
)
from ..tools.qubo_tools import QUBOContainer

class TestSmall(unittest.TestCase):
    """ Test formulations with small example """
    logger = logging.getLogger(__name__)

    def test_ab(self):
        """ Test the arc-based formulation """
        ab = get_arc_based()
        n = ab.get_num_variables()
        # Construct feasible soln:
        # D - 1 - 2 - 3 - D
        # cost:
        #   1 + 1 + 1 + 2  =  5
        x = np.zeros(n)
        d_index = ab.get_node_index("D")
        n1_index = ab.get_node_index("1")
        n2_index = ab.get_node_index("2")
        n3_index = ab.get_node_index("3")
        v_index = ab.get_var_index(d_index, 0, n1_index, 1)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1
        v_index = ab.get_var_index(n1_index, 1, n2_index, 2)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1
        v_index = ab.get_var_index(n2_index, 2, n3_index, 4)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1
        v_index = ab.get_var_index(n3_index, 4, d_index, 7)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1

        # test whether constraints are satisfied
        A_eq, b_eq, Q_eq, r_eq = ab.get_constraint_data()
        self.assertAlmostEqual(0, np.linalg.norm(A_eq.dot(x) - b_eq),
            msg="Linear constraints not satisfied"
        )
        # Not expecting any quadratic constraints
        self.assertEqual(0, Q_eq.nnz, "Too many quadratic constraints")
        self.assertEqual(0, r_eq, "Quadratic constraint data is wrong")

        # Test QUBO form
        Q, c = ab.get_qubo_new(feasibility=False)
        qubo = QUBOContainer(Q, c)
        # x should have cost 5
        self.assertEqual(5, qubo.evaluate_QUBO(x), "QUBO objective is incorrect")

    def test_pb(self):
        """ Test the path-based formulation """
        pb = get_path_based()
        n = pb.get_num_variables()
        self.assertEqual(11, n, "Number of variables is wrong")

        # Try adding infeasible routes
        feas, added = pb.add_route(['D','2','3','1','3','D'])
        self.assertFalse(feas, "Route should be infeasible")
        self.assertFalse(added, "Route should not be added")
        feas, added = pb.add_route(['D'])
        self.assertFalse(feas, "Route should be infeasible")
        self.assertFalse(added, "Route should not be added")
        # feasible, but already added
        feas, added = pb.add_route(['D','1','D'])
        self.assertTrue(feas, "Route should be feasible")
        self.assertFalse(added, "Route should not be added")

        node_name_list = pb.get_route_names(pb.routes[8])
        self.assertEqual(['D','1','2','3','D'], node_name_list, "Route is not the same")

        # Feasible solution: route D-1-2-3-D,
        # cost = 5 (see arc-based above)
        x = np.zeros(n)
        x[8] = 1
        x_infeas = np.zeros(n)
        x_infeas[8] = 1
        x_infeas[0] = 1

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

        # Test QUBO form
        Q, c = pb.get_qubo_new(feasibility=False)
        qubo = QUBOContainer(Q, c)
        # x should have cost 5
        self.assertEqual(5, qubo.evaluate_QUBO(x), "QUBO objective is incorrect")

    def test_sb(self):
        """ Test the sequence-based formulation """
        sb = get_sequence_based()
        n = sb.get_num_variables()
        # Construct INfeasible soln:
        # Not feasible because this sequence-based form is "strict"
        # D - 1 - 2 - D
        # and
        # D - 3 - D
        x = np.zeros(n)
        d_index = sb.get_node_index("D")
        n1_index = sb.get_node_index("1")
        n2_index = sb.get_node_index("2")
        n3_index = sb.get_node_index("3")
        v_index = sb.get_var_index(0, 1, n1_index,)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1
        v_index = sb.get_var_index(0, 2, n2_index,)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1
        v_index = sb.get_var_index(1, 1, n3_index,)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1
        v_index = sb.get_var_index(1, 2, d_index,)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1

        # test whether constraints are satisfied
        A_eq, b_eq, Q_eq, r_eq = sb.get_constraint_data()
        vio = max(np.linalg.norm(A_eq.dot(x) - b_eq), Q_eq.dot(x).dot(x) - r_eq)
        self.assertGreater(vio, 0, "Constraints should not be satisfied")

        # Construct INfeasible soln:
        # Not feasible because this sequence-based form is "strict"
        #       D - 1 - D
        # cost:   1 + 1 = 2
        # and
        #       D - 2 - 3 - D
        # cost:   2 + 1 + 2 = 5
        x = np.zeros(n)
        d_index = sb.get_node_index("D")
        n1_index = sb.get_node_index("1")
        n2_index = sb.get_node_index("2")
        n3_index = sb.get_node_index("3")
        v_index = sb.get_var_index(0, 1, n1_index,)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1
        v_index = sb.get_var_index(0, 2, d_index,)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1
        v_index = sb.get_var_index(1, 1, n2_index,)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1
        v_index = sb.get_var_index(1, 2, n3_index,)
        self.assertFalse(v_index is None, "Var index is wrong")
        x[v_index] = 1

        # test whether constraints are satisfied
        A_eq, b_eq, Q_eq, r_eq = sb.get_constraint_data()
        self.assertAlmostEqual(0, np.linalg.norm(A_eq.dot(x) - b_eq),
            "Linear constraints not satisfied"
        )
        self.assertAlmostEqual(0, Q_eq.dot(x).dot(x) - r_eq,
            "Quadratic constraints not satisfied"
        )

        # Test QUBO form
        Q, c = sb.get_qubo_new(feasibility=False)
        qubo = QUBOContainer(Q, c)
        # x should have cost 7
        self.assertEqual(7, qubo.evaluate_QUBO(x), "QUBO objective is incorrect")
