"""
27 January 2023
D Trenev
SM Harwood

Tools for defining and manipulating Quadratic Unconstrained Binary Optimization
(QUBO) problems
"""
import datetime
from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse as sp

# pylint: disable=invalid-name

def x_to_s(x: np.ndarray) -> np.ndarray:
    """ {0,1} to {-1,+1} variable map x :--> 1 - 2x """
    return (1 - 2 * x).astype(int)

def s_to_x(s: np.ndarray) -> np.ndarray:
    """ {-1,+1} to {0,1} variable map s :--> 0.5*(1 - s) """
    return 0.5 * (1 - s).astype(int)

def evaluate_QUBO(Q: np.ndarray, c: float, x: ArrayLike) -> float:
    """
    Evaluate the objective function of a QUBO problem
    defined by matrix/2d-array `Q`, scalar `c`,
    and vector of {0,1} `x`

    Returns `Q.dot(x).dot(x) + c`
    """
    return Q.dot(x).dot(x) + c

def evaluate_Ising(J: np.ndarray, h: ArrayLike, c: float, s: ArrayLike) -> float:
    """
    Evaluate the objective function of an Ising problem
    defined by matrix/2d-array `J`, vector `h`, scalar `c`,
    and vector of {-1,+1} ("spins") `s`

    Returns `J.dot(s).dot(s) + h.dot(s) + c`
    Note that if `J` does not have zeroed-out diagonal, this could be incorrect
    """
    return J.dot(s).dot(s) + h.dot(s) + c

def get_Ising_J_h(matrix):
    """
    Get 'J' matrix and 'h' vector from `matrix`, a `scipy.sparse` sparse array.
    Mutates `matrix` - zeroes out its diagonal
    """
    h = np.copy(matrix.diagonal())
    matrix.setdiag(0)
    return matrix, h

def QUBO_to_Ising(
        Q: ArrayLike,
        const: float=0
    ) -> Tuple[sp.csr_array, np.ndarray, float]:
    """
    Get Ising form of a QUBO.
    Consistent with {0,1} to {-1,+1} variable map
        x :--> 1 - 2x
    Uses `scipy.sparse` arrays.
    Does not modify inputs.
    """
    Q = sp.lil_array(Q)
    (n, m) = Q.shape
    if n != m:
        raise ValueError("Expected a square matrix.")
    # Convert QUBO to Ising
    J = 0.25*Q
    h = -0.25*(Q.sum(0).ravel() + Q.sum(1).ravel())
    c = 0.25*(Q.sum() + Q.diagonal().sum()) + const
    # Make the diagonal of J zero
    # This may throw a warning statement about efficiency depending on sparse type of J
    J.setdiag(0)
    J = J.tocsr()
    J.eliminate_zeros()
    return J, h, c

def Ising_to_QUBO(
        J: ArrayLike,
        h: ArrayLike,
        const: float=0
    ) -> Tuple[sp.csr_array, float]:
    """
    Get QUBO form of Ising problem.
    Consistent with {-1,+1} to {0,1} variable map
        s :--> 0.5*(1 - s)
    Uses `scipy.sparse` arrays.
    Does not modify inputs.
    """
    J = sp.csr_array(J)
    h = np.asarray(h).flatten()
    (n, m) = J.shape
    if n != m:
        raise ValueError("Expected a square matrix.")
    if n != h.shape[0]:
        raise ValueError("Expected a matrix and vector of compatible size.")
    # Convert Ising to QUBO
    Q = 4*J - 2*sp.diags(J.sum(0).ravel() + J.sum(1).ravel() + h)
    c = J.sum() + h.sum() + const
    return Q.tocsr(), c

def to_upper_triangular(M: ArrayLike) -> sp.csr_array:
    """
    Get upper triangular form `U` of problem matrix `M`:
    xᵀUx = xᵀMx.
    Returns sparse array
    """
    (n, m) = M.shape
    if n != m:
        raise ValueError("Expected a square matrix.")
    # Get strictly lower triangular part, add transpose to upper triangle,
    # then zero out lower triangle
    LT = sp.tril(M, k=-1)
    UT = sp.lil_array(M) + LT.transpose() - LT
    UT = UT.tocsr()
    UT.eliminate_zeros()
    return UT

def to_symmetric(M: ArrayLike) -> sp.csr_array:
    """
    Get symmetric form `S` of problem matrix `M`:
    xᵀSx = xᵀMx.
    Returns sparse array
    """
    (n, m) = M.shape
    if n != m:
        raise ValueError("Expected a square matrix.")
    S = sp.lil_array(M, dtype=float)
    S += S.transpose()
    S *= 0.5
    return S.tocsr()


class QUBOContainer:
    """
    Tools for defining and manipulating Quadratic Unconstrained Binary Optimization
    (QUBO) problems
    """
    def __init__(self, Q, c, pattern="upper-triangular"):
        (n, m) = Q.shape
        if n != m:
            raise ValueError("Expected a square matrix.")
        self.n_vars = n
        self.const_qubo = c
        if pattern.lower() == "upper-triangular":
            self.Q = to_upper_triangular(Q)
        elif pattern.lower() == "symmetric":
            self.Q = to_symmetric(Q)
        else:
            self.Q = sp.csr_array(Q)
        # Ising matrix has same pattern as QUBO matrix
        # (with explicitly zeroed-out diagonal)
        self.J, self.h, self.const_ising = QUBO_to_Ising(self.Q, self.const_qubo)

    def get_objective_function_QUBO(self):
        """return QUBO objective function"""
        def objective_function(x):
            return evaluate_QUBO(self.Q, self.const_qubo, x)
        return objective_function

    def get_objective_function_Ising(self):
        """return Ising objective function"""
        def objective_function(s):
            return evaluate_Ising(self.J, self.h, self.const_ising, s)
        return objective_function

    def evaluate_QUBO(self, x):
        """Evaluate QUBO objective for binaries `x`"""
        return evaluate_QUBO(self.Q, self.const_qubo, x)

    def evaluate_Ising(self, s):
        """Evaluate Ising objective for spins `s`"""
        return evaluate_Ising(self.J, self.h, self.const_ising, s)

    def report(self, obj_stats=False, tol=1e-16):
        """
        A function for generating a dictionary of 'metrics'.
        obj_stats=True will do exhaustive search over (exponentially-many) bitstrings
        """
        matrix = to_upper_triangular(self.Q)
        n = self.n_vars

        result = {}
        result["size"] = n
        nnz = matrix.nnz
        result["num_observables"] = nnz
        result["density"] = (2.0 * nnz) / ((n + 1) * n)
        # result["condition_number"] = np.linalg.cond(matrix)
        # result["distinct_eigenvalues"] = np.unique(np.linalg.eigvals(matrix)).size
        result["distinct_eigenvalues"] = np.unique(np.diagonal(matrix)).size

        if obj_stats:
            obj_funct = self.get_objective_function_QUBO()
            exp_val = 0.0
            # initialize to the objective value for [0, 0, ... ,0]
            opt_val = self.const_qubo
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
            # end loop over spins
            result["optimal_value"] = opt_val
            result["num_solutions"] = opt_count
            result["expected_value"] = exp_val
            if second_best is not None:
                result["optimality_gap"] = second_best - opt_val

        return result

    def export(self, filename=None, as_ising=False):
        """ Export the QUBO / Ising as a file with particular structure """
        N = self.n_vars
        cchar = '#'
        if as_ising:
            Mat = self.J
            d = self.h
            constant = self.const_ising
            extension = ".rudy"
        else:
            Mat = self.Q
            d = self.Q.diagonal()
            constant = self.const_qubo
            extension = ".qubo"
        contents = []
        contents.append(f"{cchar} Generated {datetime.datetime.today()}")
        contents.append(f"\n{cchar} Constant term of objective = {constant:.2f}")
        # SentinelLineIndex = len(contents)
        contents.append(f"\n{cchar} Diagonal terms")
        nDiagonals = 0
        for i in range(N):
            value = d[i]
            if value != 0:
                contents.append(f"\n{i:d} {i:d} {value: .2f}")
                nDiagonals += 1
        contents.append(f"\n{cchar} Off-Diagonal terms")
        nElements = 0
        (rows, cols, vals) = sp.find(Mat)
        for (r, c, v) in zip(rows, cols, vals):
            if r == c:
                # skip diagonal
                continue
            else:
                contents.append(f"\n{r:d} {c:d} {v: .2f}")
                nElements += 1
        # Add in program sentinel
        # sentinelLine =  '\nc Program line sentinel follows; format:'
        # sentinelLine += '\nc p qubo 0 maxDiagonals nDiagonals nElements'
        # maxDiagonals = N
        # sentinelLine += '\np qubo 0 {:d} {:d} {:d}'.format(maxDiagonals, nDiagonals, nElements)
        # contents.insert(SentinelLineIndex, sentinelLine)
        # Write to file
        if filename is None:
            filename = "fubo" + extension
        with open(filename, 'w', encoding="utf-8") as f:
            f.write("".join(contents))
        return
