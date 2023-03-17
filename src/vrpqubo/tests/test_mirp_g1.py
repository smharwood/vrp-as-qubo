"""
SM Harwood
19 October 2022
"""
import logging
from ..examples.mirp_g1 import get_mirp

logger = logging.getLogger(__name__)

def test():
    """ Test the formulations for the MIRP and solve """
    mirp = get_mirp(31)
    names = ["arc", "path", "sequence"]
    getters = [mirp.get_arc_based, mirp.get_path_based, mirp.get_sequence_based]
    logger.info(mirp)

    for name, getter in zip(names, getters):
        logger.info("\n%s-based:", name)
        r_p = getter()
        for node in r_p.nodes:
            logger.info(node)
        logger.info("Num variables: %s", r_p.get_num_variables())
        r_p.export_mip(f"ex_mirp_g1_{name}.lp")
        # r_p.solve_cplex_prob(f"ex_mirp_g1_{name}.soln")
    return
