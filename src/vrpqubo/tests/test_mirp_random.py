"""
SM Harwood
8 February 2023
"""
import logging
import unittest
from ..examples.mirp_random import get_generator
# from matplotlib import pyplot as plt

class TestMirpRandom(unittest.TestCase):
    """Subclass unittest.TestCase to test random MIRP generator"""
    logger = logging.getLogger(__name__)

    def test(self):
        """ Basic test """
        mirp_gen = get_generator(
            num_supply_ports=1,
            num_demand_ports=1,
            time_horizon=100
        )
        mirp = mirp_gen.get_random_mirp(1)
        self.logger.debug(mirp)
        self.assertEqual(len(mirp.supply_ports), 1, "Wrong number of supply ports")
        self.assertEqual(len(mirp.demand_ports), 1, "Wrong number of demand ports")
        self.assertEqual(mirp.time_horizon, 100, "Time horizon is incorrect")
        return
