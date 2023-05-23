"""
SM Harwood
23 May 2023
"""
import logging
import unittest
from ..routing_problem.vrptw import Node, Arc, VRPTW

class TestVRPTW(unittest.TestCase):
    """ Test basic VRPTW elements """
    logger = logging.getLogger(__name__)

    def test_node(self):
        """ Test Node construction """
        n = Node("foo", 1, (0,1))
        self.assertEqual("foo", n.get_name(), "Name is incorrect")
        self.assertEqual(1, n.get_demand(), "Demand is incorrect")
        self.assertEqual(-1, n.get_load(), "Load is incorrect")
        self.assertEqual((0,1), n.get_window(), "Window is incorrect")
        self.assertRaises(ValueError, Node, "bar", 0, (1,0))

    def test_arc(self):
        """ Test Arc construction """
        n1 = Node("foo", 1, (0,1))
        n2 = Node("bar", 1, (1,2))
        a = Arc(n1, n2, 1.1, 0.1)
        self.assertTrue(n1 is a.get_origin(), "Origin is incorrect")
        self.assertTrue(n2 is a.get_destination(), "Destination is incorrect")
        self.assertEqual(1.1, a.get_travel_time(), "Travel time is incorrect")
        self.assertEqual(0.1, a.get_cost(), "Cost is incorrect")

    def test_vrptw(self):
        """ Test VRPTW/graph construction """
        vrptw = VRPTW()
        # Node-related
        vrptw.add_node("foo", 1.1, (1,2))
        vrptw.add_node("bar", -1.1, (2,3))
        self.assertEqual(["foo", "bar"], vrptw.node_names, "Node name list is incorrect")
        self.assertRaises(ValueError, vrptw.add_node, "foo", 1.1, (1,2))
        vrptw.set_depot("bar")
        self.assertEqual(["bar", "foo"], vrptw.node_names, "Nodes not in correct order")
        self.assertEqual(1.1, vrptw.get_node("foo").get_demand(), "Node demand is incorrect")
        self.assertEqual(-1.1, vrptw.get_node("bar").get_demand(), "Node demand is incorrect")
        # Arc-related
        added = vrptw.add_arc("foo", "bar", 1)
        self.assertTrue(added, "Arc was not added")
        added = vrptw.add_arc("bar", "foo", 1)
        self.assertFalse(added, "Arc was added, but shouldn't have been")
        # Graph stuff
        # At this point, minimum of (incoming, outgoing) arcs at depot is zero
        self.assertEqual(0, vrptw.estimate_max_vehicles(), "Max vehicles is incorrect")
        vrptw.add_node("baz", 0, (3,4))
        vrptw.add_arc("bar", "baz", 1.1, 0)
        self.assertEqual(1, vrptw.estimate_max_vehicles(), "Max vehicles is incorrect")
