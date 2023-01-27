"""
SM Harwood
19 October 2022
"""
import logging
import numpy as np
from ..examples.small import get_path_based, get_high_cost
from ..tools.qubo_tools import QUBOContainer

logger = logging.getLogger(__name__)

def test():
    """ Test the path-based formulation with the small example """
    logger.info("Path based:")
    r_p = get_path_based()
    r_p.make_feasible(get_high_cost())
    soln = r_p.feasible_solution
    routes = r_p.get_routes(soln)
    logger.info("Routes:")
    for r in routes:
        logger.info(r)

    # Get a QUBO representation of the feasibility problem
    Q, c = r_p.get_qubo(feasibility=True)
    qubo = QUBOContainer(Q, c)

    # soln is feasible, we expect that when evaluated in the QUBO it gives
    # a zero objective
    assert 0 == qubo.evaluate_QUBO(soln), "QUBO objective not zero"

    # test whether constraints are satisfied
    A_eq, b_eq, _, _ = r_p.get_constraint_data()
    assert np.isclose(0, np.linalg.norm(A_eq.dot(soln) - b_eq)), "Linear constraints not satisfied"
    return
