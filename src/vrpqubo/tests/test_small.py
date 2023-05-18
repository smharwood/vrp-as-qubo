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
    get_high_cost
)
from ..tools.qubo_tools import QUBOContainer

class TestSmall(unittest.TestCase):
    """ Test formulations with small example """
    logger = logging.getLogger(__name__)

    def test_ab(self):
        """ Test the arc-based formulation """
        self.generic_tester(get_arc_based, "Arc based")

    def test_pb(self):
        """ Test the path-based formulation """
        self.generic_tester(get_path_based, "Path based")

    def test_sb(self):
        """ Test the sequence-based formulation """
        self.generic_tester(get_sequence_based, "Sequence based")

    def generic_tester(self, getter, form_name):
        """ Generic tester, given inputs"""
        self.logger.info("%s:", form_name)
        r_p = getter()
        r_p.make_feasible(get_high_cost())
        soln = r_p.feasible_solution
        routes = r_p.get_routes(soln)
        self.logger.debug("Routes:")
        for route in routes:
            self.logger.debug(route)

        # Get a QUBO representation of the feasibility problem
        Q, c = r_p.get_qubo(feasibility=True)
        qubo = QUBOContainer(Q, c)

        # soln is feasible, we expect that when evaluated in the QUBO it gives
        # a zero objective
        self.assertEqual(0, qubo.evaluate_QUBO(soln), "QUBO objective not zero")

        # test whether constraints are satisfied
        A_eq, b_eq, Q_eq, r_eq = r_p.get_constraint_data()
        self.assertAlmostEqual(0, np.linalg.norm(A_eq.dot(soln) - b_eq),
            "Linear constraints not satisfied"
        )
        self.assertAlmostEqual(0, Q_eq.dot(soln).dot(soln) - r_eq,
            "Quadratic constraints not satisfied"
        )
        return
