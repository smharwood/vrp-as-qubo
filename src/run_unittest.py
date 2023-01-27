"""
27 Jan 2023
SM Harwood

Run unit-ish tests for vrpqubo package
"""
import logging
from vrpqubo.tests import (
    test_arc_based,
    test_path_based,
    test_sequence_based,
    test_mirp_g1,
    test_mirp_g1_feas,
    test_sampler
)

logging.basicConfig(level=logging.DEBUG)

def main():
    """ Run unit test """
    # test_arc_based.test()
    test_path_based.test()
    # test_sequence_based.test()
    # test_mirp_g1.test()
    # test_mirp_g1_feas.test()
    # test_sampler.test()

if __name__ == "__main__":
    main()
