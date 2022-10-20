"""
SM Harwood
19 October 2022
"""
import os
import sys
import logging
# I feel this is a little hacky, but its robust to whatever the current working
# directory might be
sys.path.append(os.path.join(sys.path[0], ".."))
from examples.small import get_path_based, get_high_cost

logging.basicConfig(level=logging.DEBUG)

def test():
    """ Test the path-based formulation with the small example """
    r_p = get_path_based()
    r_p.make_feasible(get_high_cost())
    soln = r_p.feasible_solution
    routes = r_p.get_routes(soln)
    print("Routes:")
    for r in routes:
        print(r)
    return

if __name__ == "__main__":
    test()
