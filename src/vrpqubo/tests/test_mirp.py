"""
SM Harwood
12 July 2023
"""
import logging
import unittest
from ..applications.mirp import MIRP

class TestMIRP(unittest.TestCase):
    """ Test basic MIRP elements """
    logger = logging.getLogger(__name__)

    def test_time_window(self):
        """ Test time window construction """
        mirp = MIRP(cargo_size=1, time_horizon=1)
        # No prior visits, zero initial inventory, inventory fills at 1 unit/day:
        # a ship of size 1 can fill up at day 1,
        # and we max out the capacity at day 2
        tw = mirp.get_time_window(0, inventory_init=0, inventory_rate=1, inventory_cap=2)
        self.assertEqual((1,2), tw, "Time window is incorrect" )

    def test_add_nodes(self):
        """ Test adding nodes """
        mirp = MIRP(cargo_size=1, time_horizon=1)
        # Time horizon is not long enough; no nodes should be added
        names = mirp.add_nodes("foo", inventory_init=0, inventory_rate=1, inventory_cap=2)
        self.assertEqual(0, len(names), "Nodes added, but should not have been")

        mirp = MIRP(cargo_size=1, time_horizon=2)
        names = mirp.add_nodes("foo", inventory_init=0, inventory_rate=1, inventory_cap=2)
        self.assertEqual(1, len(names), "Wrong number of nodes added")

    def test_high_cost(self):
        """ Test estimate of high cost """
        mirp = MIRP(cargo_size=1, time_horizon=2)
        names = mirp.add_nodes("foo", inventory_init=0, inventory_rate=1, inventory_cap=2)
        # Add arc from depot with cost 10
        added = mirp.add_arc("Depot", names[0], 1, 10)
        self.assertTrue(added, "Arc not added, but should have been")

        # One node, which has a frequency cap/rate = 2 days
        # Must be visited once in the 2 day time horizon
        cost = mirp.estimate_high_cost()
        self.assertEqual(2*10*1, cost, "Estimated high cost is not correct")