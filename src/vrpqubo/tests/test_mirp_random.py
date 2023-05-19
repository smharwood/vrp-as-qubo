"""
SM Harwood
8 February 2023
"""
import logging
import unittest
from ..examples.mirp_random import get_generator, sample

class TestMirpRandom(unittest.TestCase):
    """ Test random MIRP generator"""
    logger = logging.getLogger(__name__)

    def test(self):
        """ Test sizing of Random MIRP """
        mirp_gen = get_generator(
            num_supply_ports=2,
            num_demand_ports=2,
            time_horizon=100
        )
        mirp = mirp_gen.get_random_mirp()
        self.logger.debug(mirp)
        self.assertEqual(len(mirp.supply_ports), 2, "Wrong number of supply ports")
        self.assertEqual(len(mirp.demand_ports), 2, "Wrong number of demand ports")
        self.assertEqual(mirp.time_horizon, 100, "Time horizon is incorrect")
        return

    def test_generator(self):
        """ Test generator of random MIRPs """
        mirp_gen = get_generator(
            num_supply_ports=1,
            num_demand_ports=1,
            time_horizon=100
        )
        mirps = mirp_gen.random_mirp_gen(2)
        for mirp in mirps:
            self.assertEqual(len(mirp.supply_ports), 1, "Wrong number of supply ports")
            self.assertEqual(len(mirp.demand_ports), 1, "Wrong number of demand ports")
            self.assertEqual(mirp.time_horizon, 100, "Time horizon is incorrect")

    def test_sample(self):
        """ Test `sample` function, and how it handles non-sampleables """
        s = sample([1,2,3], size=3)
        self.assertEqual(s, [1,2,3], "Sample is not correct")
        s = sample(1.0, size=1)
        self.assertEqual(s, 1.0, "Sample is not correct")
        self.assertRaises(ValueError, sample, 1.0, size=3)
