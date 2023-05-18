"""
SM Harwood
19 October 2022
"""
import logging
import unittest
from ..routing_problem.formulations.path_based_rp import get_sampled_key as sampler

class TestPathBased(unittest.TestCase):
    """ Test elements of path_based_rp """
    logger = logging.getLogger(__name__)

    def test_sampler(self):
        """ Test sampler in path_based_rp """
        test_dict = { 'a':100, 'b':101, 'c':102, 'd':10 }
        sample_counts = { k : 0 for k in test_dict.keys() }
        min_counts = { k : 0 for k in test_dict.keys() }
        N = 1000
        for _ in range(N):
            k_samp, k_min = sampler(test_dict, explore=0)
            sample_counts[k_samp] += 1
            min_counts[k_min] += 1
        self.logger.debug("Sample counts: %s", sample_counts)
        self.assertEqual(min_counts['d'], N,
            "Wrong number of counts for minimum"
        )
        # with no exploration, this should be equal
        self.assertEqual(sample_counts, min_counts,
            "Sample counts with no exploration is wrong"
        )
