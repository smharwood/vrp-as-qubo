"""
SM Harwood
19 October 2022
"""
import os
import sys
import logging
import numpy as np
# I feel this is a little hacky, but its robust to whatever the current working
# directory might be (assuming this is run as the main script)
sys.path.append(os.path.join(sys.path[0], ".."))
from examples.small import get_arc_based, get_high_cost
from QUBOTools import QUBOContainer

logging.basicConfig(level=logging.DEBUG)

def test():
    """ Test the arc-based formulation with the small example """
    logging.info("Arc based:")
    r_p = get_arc_based()
    r_p.make_feasible(get_high_cost())
    soln = r_p.feasible_solution
    routes = r_p.get_routes(soln)
    print("Routes:")
    for r in routes:
        print(r)

    # Get a QUBO representation of the feasibility problem
    Q, c = r_p.get_qubo(feasibility=True)
    qubo = QUBOContainer(Q, c)

    # soln is feasible, we expect that when evaluated in the QUBO it gives
    # a zero objective
    assert 0 == qubo.evaluate_QUBO(soln), "QUBO objective not zero"

    # test whether constraints are satisfied
    A_eq, b_eq, _, _ = r_p.get_constraint_data()
    assert np.isclose(0, np.linalg.norm(A_eq.dot(soln) - b_eq)), "Constraints not satisfied"
    return

if __name__ == "__main__":
    test()
