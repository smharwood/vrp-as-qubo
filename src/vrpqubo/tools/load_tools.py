"""
20 December 2018
SM Harwood

Some tools to assess Ising matrices
Note that these methods prefer to work with sparse matrices
"""
import numpy as np
import scipy.sparse

def load_spins(filename):
    """
    Read spins saved in textfile
    Assumes values are separated by whitespace and line breaks,
    reading left to right and top to bottom.
    """
    with open(filename, encoding="utf-8") as file:
        lines = file.readlines()
        spins = np.array([int(float(s)) for line in lines for s in line.split()], dtype=np.short)
        return spins

def load_matrix(filename, comment_char):
    """
    Load matrix defining problem, EITHER QUBO or Ising

    comment char is character at beginning of line for comment
    comment_char = 'c' expected for QUBO
    comment_char = '#' expected for Ising
    """
    # Store sparsely
    data = []
    row = []
    col = []
    constant = 0
    mat_length = None
    with open(filename, encoding="utf-8") as f:
        numrows = 0
        numcols = 0
        file_lines = f.readlines()
        for line in file_lines:
            if line[0] == comment_char:
                # comment line. Get constant of objective if possible;
                # if the line contains an equal sign the constant is after it
                split_contents = line.split('=')
                if len(split_contents) > 1:
                    constant = float(split_contents[1])
            elif line[0] == 'p':
                # 'Sentinel' line (specific to Dwave input- indicates QUBO)
                # p qubo 0 maxDiagonals nDiagonals nElements
                contents = line.split()
                mat_length = int(contents[4]) + int(contents[5])
            else:
                # "split()" splits based on arbitrary whitespace
                contents = line.split()
                if len(contents) == 2:
                    # Potentially get sizing for ISING form
                    # numvars numelements
                    mat_length = int(contents[1])
                else:
                    # row col val
                    row.append(int(contents[0]))
                    col.append(int(contents[1]))
                    data.append(float(contents[2]))
                    numrows = max(numrows,max(row))
                    numcols = max(numcols,max(col))
    # end with
    # A few sanity checks
    assert (numrows == numcols), "Input matrix not square"
    if mat_length is not None:
        assert (len(row) == mat_length), "Input matrix length discrepancy"
    # Construct and return
    sparse_matrix = scipy.sparse.coo_array((data,(row, col)))
    return sparse_matrix, constant

def load_qubo_matrix(filename):
    """ Helper for loading QUBOs """
    return load_matrix(filename, 'c')

def load_ising_matrix(filename):
    """ Helper for loading Ising problems """
    return load_matrix(filename, '#')

# def evaluateIsingObjective(matrix, constant, spins):
#     """
#     Given a vector of {-1,+1} spins/variables,
#     evaluate the objective/energy of the system:
#     constant + spins^T (Matrix diagonal) + spins^T (Matrix without diagonal)*spins 
#     """
#     (r,c) = matrix.shape
#     assert (r == c), "Matrix not square"
#     assert (r == len(spins)), "Number of spins incorrect for matrix"
    
#     # calculate Ising objective
#     diag = matrix.diagonal()
#     objective = constant + diag.dot(spins) + matrix.dot(spins).dot(spins) - sum(diag)
#     # Subtracting out sum of diagonal is equivalent and is faster than making a copy:
#     #MminusDiag = matrix.copy()
#     #MminusDiag.setdiag(0)
#     #objective = constant + diag.dot(spins) + MminusDiag.dot(spins).dot(spins)
#     return objective

# def evaluateQUBOObjective(matrix, constant, x):
#     """ Evaluate QUBO objective: x^T M x + constant """
#     (r, c) = matrix.shape
#     assert (r == c), "Matrix not square"
#     assert (r == len(x)), "Number of spins incorrect for matrix"
#     return matrix.dot(x).dot(x) + constant


# def exhaustiveSearch(matrix, constant, stopAtFeasible=False):
#     """
#     Go thru all possible spins to find min
#     """
#     numSpins = matrix.shape[0]
    
#     bestObj = np.inf
#     bestSpins = []
#     progCount = 0
#     progSpacing = (2**numSpins)/10000.0
#     for v in range(2**numSpins):
#         # get binary representation of integer with leading zeros
#         # then convert to {-1, +1} -values from {0,1}-valued
#         spins = [(2*int(s) - 1) for s in format(v, '0{}b'.format(numSpins))]
        
#         if v > progCount*progSpacing:
#             progCount = int(np.floor(v/progSpacing)+1)
#             print('\rConfigurations evaluated: {}%'.format(progCount/100.0), end='')
            
#         obj = evaluateIsingObjective(matrix, constant, spins)
#         if obj < bestObj:
#             bestObj = obj
#             bestSpins = [spins]
#         elif obj == bestObj:
#             # if we have a tie, append the spins
#             bestSpins.append(spins)
            
#         if stopAtFeasible and bestObj == 0:
#             break
        
#     return bestObj, bestSpins

# # DEPRECATED??
# def compareQuboAndIsing(exName):
#     """
#     Given a root example name exName, load the data in
#     exName.qubo and exName.rudy
#     and compare the objective functions on equivalent spins
#     to make sure that the conversion from QUBO to Ising is correct
#     """
#     QM, QC = loadQUBOMatrix(exName+'.qubo')
#     IM, IC = loadIsingMatrix(exName+'.rudy')
    
#     # we at least expect the size of the matrices to be the same
#     assert (QM.shape[0] == QM.shape[1]), "QUBO matrix not square"
#     assert (IM.shape[0] == IM.shape[1]), "Ising matrix not square"
#     assert (QM.shape == IM.shape), "QUBO and Ising matrices different sizes"
    
#     numTest = 100   # number of tests to perform
#     N = QM.shape[0] # number of variables
    
#     for k in range(numTest):
#         # random binary variables and corresponding "spins"
#         randomX = np.random.randint(0,2,N)
#         randomS = 2*randomX - 1
#         # calculate QUBO objective
#         objectiveQ = QC + QM.dot(randomX).dot(randomX)
#         # calculate Ising objective
#         objectiveI = evaluateIsingObjective(IM, IC, randomS)
#         assert objectiveQ == objectiveI, "Test FAILED"
#     print('Test passed!')
#     return

# def test():
#     """ Test for QUBOTools and TestTools """
#     max_err = 0
#     # Randomize a QUBO matrix
#     N = np.random.randint(5, 16)
#     Q = 100 * np.random.rand() * np.random.rand(N, N)
#     const = 10 * np.random.rand()
#     for _ in np.arange(5):
#         # Randomize a binary vector
#         x = (np.random.rand(N)[:] > 0.5).astype(int)

#         # Evaluate QUBO objective (x'Ax)
#         resultQ = QT.evaluate_QUBO(Q, const, x)

#         # Convert to Ising formulation
#         (J, h, c) = QT.QUBO_to_Ising(Q, const)
#         # Evaluate Ising objective
#         resultI = QT.evaluate_Ising(J, h, c, QT.x_to_s(x))

#         # Revert back to QUBO
#         (B, d) = QT.Ising_to_QUBO(J, h, c)
#         # Re-evaluate QUBO objective
#         resultQ2 = QT.evaluate_QUBO(B, d, x)

#         err = max(abs(resultQ - resultI), abs(resultQ - resultQ2))
#         if err < 1e-12:
#             print("Test PASSED.\r")
#         else:
#             print("Test FAILED.\r")
#         max_err = err if (err > max_err) else max_err
#     print(" Maximum error: ", max_err)

#     # Create a QUBO container
#     qb = QT.QUBOContainer(Q, const)
#     qb_obj = qb.get_objective_function_QUBO()
#     is_obj = qb.get_objective_function_Ising()
#     s = 1 - 2 * ((np.random.rand(N)[:] > 0.5).astype(int))
#     x = QT.s_to_x(s)
#     err = abs(qb_obj(x) - is_obj(s))
#     if err < 1e-12:
#         print("Test PASSED.\r")
#     else:
#         print("Test FAILED.\r")
#     print(" Objective function discrepancy: ", max_err)

#     # Play with different patterns
#     qbOG = QT.QUBOContainer(Q, const, pattern="None")
#     qbSymm = QT.QUBOContainer(Q, const, pattern="symmetric")
#     err = abs(qbOG.evaluate_QUBO(x) - qbSymm.evaluate_QUBO(x)) + abs(
#         qbOG.evaluate_QUBO(x) - qb_obj(x))
#     if err < 1e-12:
#         print("Test PASSED.\r")
#     else:
#         print("Test FAILED.\r")
#     print(" Objective function discrepancy: ", max_err)

#     # Print out objective function statistics / QUBO metrics for our usual test problem.
#     matrix, constant = loadQUBOMatrix('path_based/ExSmall.qubo')
#     qb = QT.QUBOContainer(matrix, constant)
#     print("Metrics for test problem:", qb.report(True))


# if __name__ == "__main__":
#     main()