"""
SM Harwood
23 May 2023
"""
import logging
import unittest
from ..routing_problem.routing_problem import RoutingProblem
from ..routing_problem.vrptw import VRPTW

class TestRoutingProblem(unittest.TestCase):
    """ Test basic RoutingProblem elements """
    logger = logging.getLogger(__name__)

    def test_graph(self):
        """ Test logic around copying graph """
        vrptw = VRPTW()
        vrptw.add_node("foo", 1)
        rp = RoutingProblem(vrptw)
        # Should be a deep copy of the objects
        self.assertFalse(vrptw is rp.vrptw, "VRPTW object was not copied")
        self.assertEqual(vrptw.node_names, rp.vrptw.node_names, "VRPTW objects are incorrect")
