"""
SM Harwood
8 February 2023
"""
import logging
from ..examples.mirp_random import get_generator
# from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

def test():
    """ Test the random MIRP generator """
    mirp_gen = get_generator(num_supply_ports=1, num_demand_ports=1, time_horizon=100)
    mirp = mirp_gen.get_random_mirp(1)
    logger.info(mirp)
    return
