"""
SM Harwood
19 October 2022
"""
import sys
import numpy as np
sys.path.append("../")
from examples.mirp_g1 import get_mirp

def test():
    """ Test the feasibility heuristic, and constraint data """
    mirp = get_mirp(300)
    names = ["arc", "path", "sequence"]
    getters = [mirp.get_arc_based, mirp.get_path_based, mirp.get_sequence_based]

    for name, getter in zip(names, getters):
        r_p = getter(make_feasible=True)
        f_s = r_p.feasible_solution
        A_eq, b_eq, Q_eq, r_eq = r_p.get_constraint_data()
        res = A_eq.dot(f_s) - b_eq
        res_q = Q_eq.dot(f_s).dot(f_s) - r_eq

        print(f"{name}-based residual: {res}")
        assert np.isclose(np.linalg.norm(res), 0), f"{name}-based feasible solution is not feasible"
        assert np.isclose(res_q, 0), f"{name}-based feasible solution is not feasible (quad con)"
    return

if __name__ == "__main__":
    test()
