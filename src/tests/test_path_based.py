"""
SM Harwood
19 October 2022
"""
import sys
import logging
sys.path.append("..")
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
