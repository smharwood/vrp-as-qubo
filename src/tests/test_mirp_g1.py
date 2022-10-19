"""
SM Harwood
19 October 2022
"""
import sys
import logging
sys.path.append("..")
from examples.mirp_g1 import get_mirp

logging.basicConfig(level=logging.DEBUG)

def test():
    """ Test the formulations for the MIRP and solve """
    mirp = get_mirp(31)
    names = ["arc", "path", "sequence"]
    getters = [mirp.get_arc_based, mirp.get_path_based, mirp.get_sequence_based]

    for name, getter in zip(names, getters):
        print(f"\n{name}-based:")
        r_p = getter()
        for node in r_p.nodes:
            print(node)
        print(f"Num variables: {r_p.get_num_variables()}")
        r_p.export_mip(f"ex_mirp_g1_{name}.lp")
        # r_p.solve_cplex_prob(f"ex_mirp_g1_{name}.soln")
    return

if __name__ == "__main__":
    test()
