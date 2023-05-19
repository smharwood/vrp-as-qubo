"""
SM Harwood
19 October 2022
"""
import logging
import unittest
from functools import partial
import numpy as np
from ..examples.mirp_g1 import get_mirp

class TestMirpG1(unittest.TestCase):
    """ Test the formulations for the mirp_g1 example (and solve?) """
    logger = logging.getLogger(__name__)

    def setUp(self) -> None:
        """ Construct common MIRP example """
        self.mirp = get_mirp(31)
        # prolly need to punch time horizon up to 300
        # to get to truly infeasible instances
        self.big_mirp = get_mirp(100)
        self.logger.debug(self.mirp)
        return super().setUp()

    def test_ab(self):
        """Test arc-based formulation sizing"""
        name = "arc"
        size = 1680
        getter = self.mirp.get_arc_based
        self.generic_sizing_tester(getter, name, size)

    def test_ab_feas(self):
        """Test arc-based formulation feasibility"""
        name = "arc"
        getter = self.big_mirp.get_arc_based
        self.generic_feas_tester(getter, name)

    def test_pb(self):
        """Test path-based formulation sizing"""
        name = "path"
        size = 96
        getter = self.mirp.get_path_based
        self.generic_sizing_tester(getter, name, size)

    def test_pb_feas(self):
        """Test path-based formulation feasibility"""
        name = "path"
        getter = self.big_mirp.get_path_based
        self.generic_feas_tester(getter, name)

    def test_sb(self):
        """Test sequence-based formulation sizing"""
        name = "sequence"
        size = 900
        getter = self.mirp.get_sequence_based
        self.generic_sizing_tester(getter, name, size)

    def test_sb_feas(self):
        """Test sequence-based formulation feasibility"""
        name = "sequence"
        getter = partial(self.big_mirp.get_sequence_based, strict=False)
        self.generic_feas_tester(getter, name)

    def generic_sizing_tester(self, getter, name, size):
        """ Test the sizing of the formulations """
        self.logger.info("\n%s-based:", name)
        r_p = getter()
        for node in r_p.nodes:
            self.logger.debug(node)
        self.assertEqual(size, r_p.get_num_variables(),
            f"Unexpected number of variables for {name}"
        )
        # r_p.export_mip(f"ex_mirp_g1_{name}.lp")
        # r_p.solve_cplex_prob(f"ex_mirp_g1_{name}.soln")
        return

    def generic_feas_tester(self, getter, name):
        """ Test the feasibility heuristic, and constraint data """
        self.logger.info("\n%s-based:", name)
        r_p = getter(make_feasible=True)
        f_s = r_p.feasible_solution
        A_eq, b_eq, Q_eq, r_eq = r_p.get_constraint_data()
        res = A_eq.dot(f_s) - b_eq
        res_q = Q_eq.dot(f_s).dot(f_s) - r_eq

        self.logger.debug("%s-based residual: %s", name, res)
        self.assertAlmostEqual(0, np.linalg.norm(res),
            f"{name}-based feasible solution is not feasible"
        )
        self.assertAlmostEqual(0, res_q,
            f"{name}-based feasible solution is not feasible (quad con)"
        )
        return
