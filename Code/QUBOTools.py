import datetime
import numpy as np
import scipy.sparse as sp

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