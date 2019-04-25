A test set of vehicle routing problems as Ising models

File 
    test_m_n_c.rudy
contains a sparse representation of the Ising matrix/vector for a certain test problem, where
    m : indicates the number of constraints in the original integer programming (IP) formulation
    n : indicates the number of variables or spins required
    c : indicates the problem "class"
        = f : a feasibility problem (minimum value is 0)
        = o : an optimization problem
    
The .mps files contain a specification of the problem in its original IP form.
The MPS format is standard and can be read by many solvers.

TestTools.py contains some tools to read the Ising matrix and evaluate the objective for given spins.
For example,
    $python TestTools.py test_14_13_f.rudy optimalSpins13.txt
will load the matrix and spins indicated in the files, visualize the sparsity of the matrix, and evaluate the objective for these spins.

If no spins file is given, then minimization is performed:
    $python TestTools.py test_14_13_f.rudy
will load the matrix, visualize the sparsity, AND find minimizing spins by EXHAUSTIVE search (so don't use it unless the problem is small!)

Details:
test_14_13_f    :   feasible instance (minimum = 0)
test_14_14_f    :   feasible instance (minimum = 0)
test_14_14_o    :   feasible instance, nonzero costs (minimum = 40785)
                    Same constraints as test_14_14_f
test_17_17_f    : INfeasible instance (minimum = 1.0)
test_17_26_f    :   feasible instance (minimum = 0)
                    Same constraints as test_17_17_f, with more variables
test_17_26_o    :   feasible instance, nonzero costs (minimum = 50548)
                    Same constraints as test_17_26_f
test_21_28_f    :   feasible instance (minimum = 0)
test_21_49_f    :   feasible instance (minimum = 0)
                    Same constraints as test_21_28_f, with more variables
test_21_56_f    :   feasible instance (minimum = 0)
test_21_56_o    :   feasible instance, nonzero costs (minimum = 68322)
                    Same constraints as test_21_56_f