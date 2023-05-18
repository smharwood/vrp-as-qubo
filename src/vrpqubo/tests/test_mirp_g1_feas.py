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
    # logging.basicConfig(level = logging.INFO)

    def test_sizing(self):
        """ Test the sizing of the formulations """
        mirp = get_mirp(31)
        names = ["arc", "path", "sequence"]
        getters = [mirp.get_arc_based,
                   mirp.get_path_based,
                   mirp.get_sequence_based
                ]
        sizing = [1680, 96, 900]
        self.logger.debug(mirp)

        for name, getter, size in zip(names, getters, sizing):
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

    def ntest_feas(self):
        """ Test the feasibility heuristic, and constraint data """
        mirp = get_mirp(300)
        names= []
        getters = []
        names.append("arc")
        getters.append(mirp.get_arc_based)
        names.append("path")
        getters.append(mirp.get_path_based)
        names.append("sequence")
        getters.append(partial(mirp.get_sequence_based, strict=False))

        for name, getter in zip(names, getters):
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
