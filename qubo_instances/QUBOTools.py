import numpy as np
import scipy as sp
from scipy import sparse

# inline functions for converting numpy arrays(!) of binary values {0,1} to/from {1,-1}
x_to_s = lambda x: (1 - 2 * x).astype(int)
s_to_x = lambda s: 0.5 * (1 - s).astype(int)


# objective functions for general QUBO (Q - matrix, c - scalar, x - (binary) vector)
def evaluate_QUBO(Q, c, x):
    return Q.dot(x).dot(x) + c


# objective function for general Ising (J - matrix, h - vector, c - scalar, s - vector (of {1,-1}))
def evaluate_Ising(J, h, c, s):
    return J.dot(s).dot(s) + h.dot(s) + c


# Conversion functions

# convert input to type numpy.ndarray for use with the functions here.
def get_np_array(matrix):
    result = matrix
    if type(result) is not np.ndarray:
        try:
            result = result.toarray()
        except:
            print(type(matrix), " can not be converted to a (numpy.ndarray) matrix.")
            raise
    return result


# QUBO -> Ising
def QUBO_to_Ising(Q, const=0):
    # Work with numpy arrays
    matrix = get_np_array(Q)
    (n, m) = matrix.shape
    assert (n == m), "Expected a square matrix."
    # Convert QUBO to Ising
    J = matrix * 0.25
    h = -0.25 * matrix.sum(0) - 0.25 * matrix.sum(1)
    c = 0.25 * matrix.sum() + const
    c += np.diag(J).sum()
    # Make the diagonal of J zero
    J -= np.diag(np.diag(J))
    return (J, h, c)


# Ising -> QUBO
def Ising_to_QUBO(J, h, const=0):
    # Work with numpy arrays
    matrix = get_np_array(J)
    vector = get_np_array(h).flatten()
    (n, m) = matrix.shape
    assert ((n == m) and (
                n == vector.shape[0])), "Expected a square matrix and a vector of compatible size."
    # Convert Ising to QUBO
    Q = matrix * 4 - 2 * np.diag(matrix.sum(0) + matrix.sum(1) + vector)
    c = matrix.sum() + vector.sum() + const
    return (Q, c)


# Upper-triangular version
def to_upper_triangular(M):
    # Work with numpy arrays
    matrix = get_np_array(M)
    (n, m) = matrix.shape
    assert (n == m), "Expected a square matrix."
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] += matrix[j][i]
            matrix[j][i] = 0.0
    return matrix


# Symmetric version
def to_symmetric(M):
    # Work with numpy arrays
    matrix = np.array(M)
    (n, m) = matrix.shape
    assert (n == m), "Expected a square matrix."
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = 0.5 * (matrix[j][i] + matrix[i][j])
            matrix[j][i] = matrix[i][j]
    return matrix


class QUBOContainer:

    def __init__(self, Q, c, pattern="upper-triangular"):
        self.Q = get_np_array(Q)
        (n, m) = self.Q.shape
        assert (n == m), "Expected a square matrix."
        self.numVars = n
        self.constQ = c
        (self.J, self.h, self.constI) = QUBO_to_Ising(self.Q, self.constQ)
        if pattern.lower() == "upper-triangular":
            self.Q = to_upper_triangular(self.Q)
            self.J = to_upper_triangular(self.J)
        elif pattern.lower() == "symmetric":
            self.Q = to_symmetric(self.Q)
            self.J = to_symmetric(self.J)

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
        nnz = np.count_nonzero(matrix)
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

    def export(self, filename=None):
        """ Export the QUBO as a file with particular structure """
        N = self.numVars
        contents = []
        # contents.append('c generated {}\n'.format(datetime.datetime.today()))
        contents.append('c Constant term of objective = {:.2f}'.format(self.constQ))
        contents.append('\nc Program line sentinel follows; format:')
        contents.append('\nc p qubo 0 maxDiagonals nDiagonals nElements')
        # Don't have these data counts yet. Insert later
        SentinelLineIndex = len(contents)
        contents.append('\nc Diagonal terms')
        maxDiagonals = N
        nDiagonals = 0
        d = self.Q.diagonal()
        for i in range(N):
            value = d[i]
            if value != 0:
                contents.append('\n{:d} {:d} {: .2f}'.format(i, i, value))
                nDiagonals += 1
        contents.append('\nc Off-Diagonal terms')
        nElements = 0
        (rows, cols) = np.nonzero(self.Q)
        vals = self.Q[(rows, cols)]
        for (r, c, v) in zip(rows, cols, vals):
            if r == c:
                # skip diagonal
                continue
            else:
                contents.append('\n{:d} {:d} {: .2f}'.format(r, c, v))
                nElements += 1
        # Add in program sentinel
        sentinelLine = '\np qubo 0 {:d} {:d} {:d}'.format(maxDiagonals, nDiagonals, nElements)
        contents.insert(SentinelLineIndex, sentinelLine)
        # Write to file
        if filename is None:
            filename = 'fubo.qubo'
        f = open(filename, 'w')
        f.write("".join(contents))
        f.close()
        return


if __name__ == '__main__':
    max_err = 0
    # Randomize a QUBO matrix
    N = np.random.randint(5, 16)
    Q = 100 * np.random.rand() * np.random.rand(N, N)
    const = 10 * np.random.rand()
    for i in np.arange(5):
        # Randomize a binary vector
        x = (np.random.rand(N)[:] > 0.5).astype(int)

        # Evaluate QUBO objective (x'Ax)
        resultQ = evaluate_QUBO(Q, const, x)

        # Convert to Ising formulation
        (J, h, c) = QUBO_to_Ising(Q, const)
        # Evaluate Ising objective
        resultI = evaluate_Ising(J, h, c, x_to_s(x))

        # Revert back to QUBO
        (B, d) = Ising_to_QUBO(J, h, c)
        # Re-evaluate QUBO objective
        resultQ2 = evaluate_QUBO(B, d, x)

        err = max(abs(resultQ - resultI), abs(resultQ - resultQ2))
        if err < 1e-12:
            print("Test PASSED.\r")
        else:
            print("Test FAILED.\r")
        max_err = err if (err > max_err) else max_err
    print(" Maximum error: ", max_err)

    # Create a QUBO container
    qb = QUBOContainer(Q, const)
    qb_obj = qb.get_objective_function_QUBO()
    is_obj = qb.get_objective_function_Ising()
    s = 1 - 2 * ((np.random.rand(N)[:] > 0.5).astype(int))
    x = s_to_x(s)
    err = abs(qb_obj(x) - is_obj(s))
    if err < 1e-12:
        print("Test PASSED.\r")
    else:
        print("Test FAILED.\r")
    print(" Objective function discrepancy: ", max_err)

    # Play with different patterns
    qbOG = QUBOContainer(Q, const, pattern="None")
    qbSymm = QUBOContainer(Q, const, pattern="symmetric")
    err = abs(qbOG.evaluate_QUBO(x) - qbSymm.evaluate_QUBO(x)) + abs(
        qbOG.evaluate_QUBO(x) - qb_obj(x))
    if err < 1e-12:
        print("Test PASSED.\r")
    else:
        print("Test FAILED.\r")
    print(" Objective function discrepancy: ", max_err)

    # Print out objective function statistics / QUBO metrics for our usual test problem.
    from path_based.TestTools import loadQUBOMatrix

    matrix, constant = loadQUBOMatrix('path_based/ExSmall.qubo')
    qb = QUBOContainer(matrix, constant)
    print("Metrics for test problem:", qb.report(True))

