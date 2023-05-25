"""
30 September 2019
SM Harwood

Test instances for Stanford/Cornell:
vary TimeHorizon and number of routes added to get different size problems
"""
import os
import argparse
from functools import partial
import numpy as np
from vrpqubo.tools.qubo_tools import QUBOContainer, x_to_s
from vrpqubo.examples.mirp_g1 import get_mirp
try:
    import cplex
    from solve_w_cplex import CPLEX_FEASIBLE
    HAVE_CPLEX = True
except ImportError:
    HAVE_CPLEX = False

def main():
    """ Build a set of test problems """
    parser = argparse.ArgumentParser(description=
        "Build a set of test problems\n\n"+
        "For example, running\n"+
        "python generate_test_set.py -p TestSet -t 20 30 40 50\n"+
        "builds a test set in the folder \"TestSet\" with 24 total instances\n"+
        "(\"feasibility\" and \"optimality\" versions for each of three different formulations\n"+
        "for each of the four time horizons given)\n\n"+
        "Each file has the name \"test_<formulation>_<size>_<class>.rudy\"\n"+
        "where <formulation> indicates which formulation is used,\n"+
        "      <size> indicates the number of variables,\n"+
        "      <class> indicates whether its a feasibility problem "+
        "(\'f\', optimal value is zero) or not (\'o\')",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p','--prefix', type=str, default='.',
                        help="Folder to put these test problem definitions")
    parser.add_argument('-t','--time_horizons', nargs='+', type=float,
                        help="Time horizons of problems to generate")
    args = parser.parse_args()
    if args.time_horizons is None:
        parser.print_help()
        return
    if not os.path.isdir(args.prefix):
        os.mkdir(args.prefix)
    gen(args.prefix, args.time_horizons)
    return

def gen(prefix, horizons):
    """ Generate test set """

    formulations = ["arc_based", "path_based", "sequence_based"]
    for t_h in horizons:
        # Define base problem
        mirp = get_mirp(t_h)
        formulation_getters = [
            mirp.get_arc_based,
            mirp.get_path_based,
            partial(mirp.get_sequence_based, strict=False)
        ]
        for form, getter in zip(formulations, formulation_getters):
            name = ''.join([w[0] for w in form.split('_')])

            # Get the Routing Problem object
            r_p = getter(make_feasible=True)

            # Version with objective:
            Q, c = r_p.get_qubo(feasibility=False)
            QC = QUBOContainer(Q, c)
            n_vars = r_p.get_num_variables()
            print(
                f"Time horizon {t_h}, "+\
                f"formulation {form}, "+\
                f"number of variables: {n_vars}"
            )
            bname = f"test_{name}_{n_vars}_"
            bname = os.path.join(prefix, bname)
            QC.export(bname + "o.rudy", as_ising=True)

            # Feasibility version
            Q, c = r_p.get_qubo(feasibility=True)
            QC = QUBOContainer(Q, c)
            QC.export(bname + "f.rudy", as_ising=True)

            # export constraint set
            # Note that some of these matrices might be scipy.sparse,
            # in which case savez is not the most natural way to save them...
            # but we can hack our way around it
            A_eq, b_eq, Q_eq, r_eq = r_p.get_constraint_data()
            np.savez(bname, A_eq=A_eq, b_eq=b_eq, Q_eq=Q_eq, r_eq=r_eq)

            if HAVE_CPLEX:
                # We should solve the problem now -
                # variable order might get messed up reading from the .lp
                # Also, focus on finding feasible solution to test other features
                r_p.export_mip(bname + "o.lp")
                cplex_prob = r_p.get_cplex_prob()
                cplex_prob.parameters.mip.limits.solutions.set(1)
                try:
                    cplex_prob.solve()
                except cplex.exceptions.CplexError as err:
                    print(f"CPLEX Error: {err}")
                    continue
                stat = cplex_prob.solution.get_status_string()
                if stat.lower() not in CPLEX_FEASIBLE:
                    print(f"No solution written; status: {stat}")
                    continue
                xstar = cplex_prob.solution.get_values()
                sol_path = bname + ".sol"
                with open(sol_path, 'w', encoding="utf-8") as spin_file:
                    spins = x_to_s(np.asarray(xstar))
                    for spin in spins:
                        spin_file.write(f"{int(spin)}\n")
    return

if __name__ == "__main__":
    main()
