import datetime
import numpy as np
import scipy.sparse as sp
# from TestTools import loadQUBOMatrix


# inline functions for converting numpy arrays(!) of binary values {0,1} to/from {1,-1}
x_to_s = lambda x: (1 - 2 * x).astype(int)
s_to_x = lambda s: 0.5 * (1 - s).astype(int)

def evaluate_QUBO(Q, c, x):
    """ Evaluate the objective function of a QUBO problem
    defined by matrix/2d-array Q, scalar c,
    and vector of {0,1} x

    Returns Q.dot(x).dot(x) + c
    """
    return Q.dot(x).dot(x) + c

def evaluate_Ising(J, h, c, s):
    """ Evaluate the objective function of an Ising problem
    defined by matrix/2d-array J, vector h, scalar c,
    and vector of {-1,+1} ("spins") s

    Returns J.dot(s).dot(s) + h.dot(s) + c
    Note that if J does not have zeroed-out diagonal, this could be incorrect
    """
    return J.dot(s).dot(s) + h.dot(s) + c

# Conversion functions

# # convert input to type numpy.ndarray for use with the functions here.
# def get_np_array(matrix):
#     result = matrix
#     if type(result) is not np.ndarray:
#         try:
#             result = result.toarray()
#         except:
#             print(type(matrix), " can not be converted to a (numpy.ndarray) matrix.")
#             raise
#     return result

def get_Ising_J_h(matrix):
    """ Get 'J' matrix and 'h' vector from matrix 
    Mutates `matrix` - zeroes out its diagonal
    """
    h = np.copy(matrix.diagonal())
    matrix.setdiag(0)
    return matrix, h

# QUBO -> Ising
# def QUBO_to_Ising(Q, const=0):
#     # Work with numpy arrays
#     matrix = get_np_array(Q)
#     (n, m) = matrix.shape
#     assert (n == m), "Expected a square matrix."
#     # Convert QUBO to Ising
#     J = matrix * 0.25
#     h = -0.25 * matrix.sum(0) - 0.25 * matrix.sum(1)
#     c = 0.25 * matrix.sum() + const
#     c += np.diag(J).sum()
#     # Make the diagonal of J zero
#     J -= np.diag(np.diag(J))
#     return (J, h, c)
def QUBO_to_Ising(Q, const=0):
    """ Get Ising form of a QUBO
    Consistent with {0,1} to {-1,+1} variable map 
        x :--> 1 - 2x
    Uses scipy.sparse arrays
    Does not modify inputs
    """
    Q = sp.lil_matrix(Q)
    (n, m) = Q.shape
    assert (n == m), "Expected a square matrix."
    # Convert QUBO to Ising
    J = 0.25*Q
    h = -0.25*(Q.sum(0).A1 + Q.sum(1).A1)
    c = 0.25*(Q.sum() + Q.diagonal().sum()) + const
    # Make the diagonal of J zero
    # This may throw a warning statement about efficiency depending on sparse type of J
    J.setdiag(0)
    J = J.tocsr()
    J.eliminate_zeros()
    return (J, h, c)

# Ising -> QUBO
# def Ising_to_QUBO(J, h, const=0):
#     # Work with numpy arrays
#     matrix = get_np_array(J)
#     vector = get_np_array(h).flatten()
#     (n, m) = matrix.shape
#     assert ((n == m) and (
#                 n == vector.shape[0])), "Expected a square matrix and a vector of compatible size."
#     # Convert Ising to QUBO
#     Q = matrix * 4 - 2 * np.diag(matrix.sum(0) + matrix.sum(1) + vector)
#     c = matrix.sum() + vector.sum() + const
#     return (Q, c)
def Ising_to_QUBO(J, h, const=0):
    """ Get QUBO form of Ising problem
    Consistent with {-1,+1} to {0,1} variable map 
        s :--> 0.5*(1 - s)
    Uses scipy.sparse arrays
    Does not modify inputs
    """
    J = sp.csr_matrix(J)
    h = np.asarray(h).flatten()
    (n, m) = J.shape
    assert ((n == m) and 
            (n == h.shape[0])), "Expected a square matrix and a vector of compatible size."
    # Convert Ising to QUBO
    # Note: scipy.sparse.matrix.sum() returns a numpy matrix,
    #       and attribute .A1 is the flattened ndarray
    Q = 4*J - 2*sp.diags(J.sum(0).A1 + J.sum(1).A1 + h)
    c = J.sum() + h.sum() + const
    return (Q.tocsr(), c)

# Upper-triangular version
# def to_upper_triangular(M):
#     # Work with numpy arrays
#     matrix = get_np_array(M)
#     (n, m) = matrix.shape
#     assert (n == m), "Expected a square matrix."
#     for i in range(n):
#         for j in range(i + 1, n):
#             matrix[i][j] += matrix[j][i]
#             matrix[j][i] = 0.0
#     return matrix
def to_upper_triangular(M):
    """ Get upper triangular form of problem matrix
    Returns sparse matrix
    """
    (n, m) = M.shape
    assert (n == m), "Expected a square matrix."
    # Get strictly lower triangular part, add transpose to upper triangle,
    # then zero out lower triangle
    LT = sp.tril(M, k=-1)
    UT = sp.lil_matrix(M) + LT.transpose() - LT
    UT = UT.tocsr()
    UT.eliminate_zeros()
    return UT

# Symmetric version
# def to_symmetric(M):
#     # Work with numpy arrays
#     matrix = np.array(M)
#     (n, m) = matrix.shape
#     assert (n == m), "Expected a square matrix."
#     for i in range(n):
#         for j in range(i + 1, n):
#             matrix[i][j] = 0.5 * (matrix[j][i] + matrix[i][j])
#             matrix[j][i] = matrix[i][j]
#     return matrix
def to_symmetric(M):
    """ Get symmetric form of problem matrix
    Returns sparse matrix
    """
    (n, m) = M.shape
    assert (n == m), "Expected a square matrix."
    S = sp.lil_matrix(M)
    S += S.transpose()
    S *= 0.5
    return S.tocsr()


class QUBOContainer:

    def __init__(self, Q, c, pattern="upper-triangular"):
        (n, m) = Q.shape
        assert (n == m), "Expected a square matrix."
        self.numVars = n
        self.constQ = c
        if pattern.lower() == "upper-triangular":
            self.Q = to_upper_triangular(Q)
        elif pattern.lower() == "symmetric":
            self.Q = to_symmetric(Q)
        else:
            self.Q = sp.csr_matrix(Q)
        # Ising matrix has same pattern as QUBO matrix
        # (with explicitly zeroed-out diagonal)
        (self.J, self.h, self.constI) = QUBO_to_Ising(self.Q, self.constQ)

    # def __init__(self, J, h, c):
    #    self.J = get_np_array(J)
    #    self.h = get_np_array(h).flatten()
    #    (n,m) = self.J.shape
    #    assert ((n == m) and (n==self.h.shape[0])), "Expected a square matrix and a vector of compatible size."
    #    self.numVars = n
    #    self.constI = c
    #    (self.Q, self.constQ) = Ising_to_QUBO(self.J, self.h, self.constI)

    def get_objective_function_QUBO(self):
        return lambda x: self.Q.dot(x).dot(x) + self.constQ

    def get_objective_function_Ising(self):
        return lambda s: self.J.dot(s).dot(s) + self.h.dot(s) + self.constI

    def evaluate_QUBO(self, x):
        return evaluate_QUBO(self.Q, self.constQ, x)

    def evaluate_Ising(self, s):
        return evaluate_Ising(self.J, self.h, self.constI, s)

    # A function for generating a dictionary of "metrics".
    def report(self, includeObjectiveStats=False, tol=1e-16):
        matrix = to_upper_triangular(self.Q)
        n = self.numVars

        result = {}
        result["size"] = n
        nnz = matrix.nnz
        result["num_observables"] = nnz
        result["density"] = (2.0 * nnz) / ((n + 1) * n)
        # result["condition_number"] = np.linalg.cond(matrix)
        # result["distinct_eigenvalues"] = np.unique(np.linalg.eigvals(matrix)).size
        result["distinct_eigenvalues"] = np.unique(np.diagonal(matrix)).size

        if includeObjectiveStats:
            obj_funct = self.get_objective_function_QUBO()
            exp_val = 0.0
            opt_val = self.constQ  # initialize to the objective value for [0, 0, ... ,0]
            second_best = None
            opt_count = 1
            N = 2 ** n
            for v in range(1, N):
                x = [int(s) for s in format(v, '0{}b'.format(n))]
                obj_val = obj_funct(x)
                exp_val += obj_val / N
                if abs(obj_val - opt_val) <= tol:
                    opt_count += 1
                elif obj_val < opt_val:
                    second_best = opt_val
                    opt_val = obj_val
                    opt_count = 1
                elif second_best is None:
                    second_best = obj_val
            result["optimal_value"] = opt_val
            result["num_solutions"] = opt_count
            result["expected_value"] = exp_val
            if second_best is not None:
                result["optimality_gap"] = second_best - opt_val

        return result

    def export(self, filename=None, as_ising=False):
        """ Export the QUBO / Ising as a file with particular structure """
        N = self.numVars
        cchar = '#'
        if as_ising:
            Mat = self.J
            d = self.h
            constant = self.constI
            extension = '.rudy'
        else:
            Mat = self.Q
            d = self.Q.diagonal()
            constant = self.constQ
            extension = '.qubo'
        contents = []
        contents.append('{} Generated {}'.format(cchar, datetime.datetime.today()))
        contents.append('\n{} Constant term of objective = {:.2f}'.format(cchar, constant))
        # SentinelLineIndex = len(contents)
        contents.append('\n{} Diagonal terms'.format(cchar))
        nDiagonals = 0
        for i in range(N):
            value = d[i]
            if value != 0:
                contents.append('\n{:d} {:d} {: .2f}'.format(i, i, value))
                nDiagonals += 1
        contents.append('\n{} Off-Diagonal terms'.format(cchar))
        nElements = 0
        (rows, cols, vals) = sp.find(Mat)
        for (r, c, v) in zip(rows, cols, vals):
            if r == c:
                # skip diagonal
                continue
            else:
                contents.append('\n{:d} {:d} {: .2f}'.format(r, c, v))
                nElements += 1
        # Add in program sentinel
        # sentinelLine =  '\nc Program line sentinel follows; format:'
        # sentinelLine += '\nc p qubo 0 maxDiagonals nDiagonals nElements'
        # maxDiagonals = N
        # sentinelLine += '\np qubo 0 {:d} {:d} {:d}'.format(maxDiagonals, nDiagonals, nElements)
        # contents.insert(SentinelLineIndex, sentinelLine)
        # Write to file
        if filename is None:
            filename = 'fubo' + extension
        with open(filename, 'w') as f:
            f.write("".join(contents))
        return


# if __name__ == '__main__':
#     max_err = 0
#     # Randomize a QUBO matrix
#     N = np.random.randint(5, 16)
#     Q = 100 * np.random.rand() * np.random.rand(N, N)
#     const = 10 * np.random.rand()
#     for i in np.arange(5):
#         # Randomize a binary vector
#         x = (np.random.rand(N)[:] > 0.5).astype(int)

#         # Evaluate QUBO objective (x'Ax)
#         resultQ = evaluate_QUBO(Q, const, x)

#         # Convert to Ising formulation
#         (J, h, c) = QUBO_to_Ising(Q, const)
#         # Evaluate Ising objective
#         resultI = evaluate_Ising(J, h, c, x_to_s(x))

#         # Revert back to QUBO
#         (B, d) = Ising_to_QUBO(J, h, c)
#         # Re-evaluate QUBO objective
#         resultQ2 = evaluate_QUBO(B, d, x)

#         err = max(abs(resultQ - resultI), abs(resultQ - resultQ2))
#         if err < 1e-12:
#             print("Test PASSED.\r")
#         else:
#             print("Test FAILED.\r")
#         max_err = err if (err > max_err) else max_err
#     print(" Maximum error: ", max_err)

#     # Create a QUBO container
#     qb = QUBOContainer(Q, const)
#     qb_obj = qb.get_objective_function_QUBO()
#     is_obj = qb.get_objective_function_Ising()
#     s = 1 - 2 * ((np.random.rand(N)[:] > 0.5).astype(int))
#     x = s_to_x(s)
#     err = abs(qb_obj(x) - is_obj(s))
#     if err < 1e-12:
#         print("Test PASSED.\r")
#     else:
#         print("Test FAILED.\r")
#     print(" Objective function discrepancy: ", max_err)

#     # Play with different patterns
#     qbOG = QUBOContainer(Q, const, pattern="None")
#     qbSymm = QUBOContainer(Q, const, pattern="symmetric")
#     err = abs(qbOG.evaluate_QUBO(x) - qbSymm.evaluate_QUBO(x)) + abs(
#         qbOG.evaluate_QUBO(x) - qb_obj(x))
#     if err < 1e-12:
#         print("Test PASSED.\r")
#     else:
#         print("Test FAILED.\r")
#     print(" Objective function discrepancy: ", max_err)

#     # Print out objective function statistics / QUBO metrics for our usual test problem.
#     matrix, constant = loadQUBOMatrix('path_based/ExSmall.qubo')
#     qb = QUBOContainer(matrix, constant)
#     print("Metrics for test problem:", qb.report(True))