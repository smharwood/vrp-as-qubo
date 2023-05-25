"""
25 May 2023
SM Harwood
"""
import logging
import unittest
import numpy as np
from scipy import sparse
from ..tools import qubo_tools

class TestQUBOTools(unittest.TestCase):
    """ Test QUBO Tools """
    logger = logging.getLogger(__name__)

    def test_converters(self):
        """Test converting spins and binaries"""
        # Note that these mappings are arbitrary, but it is important to be consistent
        x = np.array([1,0,1,0,1,0])
        s = qubo_tools.x_to_s(x)
        self.assertEqual(s.tolist(), [-1, 1, -1, 1, -1, 1], "Spins are not correct")
        # Roundtrip
        x2 = qubo_tools.s_to_x(s)
        self.assertEqual(np.linalg.norm(x-x2), 0, "Roundtrip binaries not equal")

        s = np.array([-1, 1, -1, 1, -1, 1])
        x = qubo_tools.s_to_x(s)
        self.assertEqual(x.tolist(), [1,0,1,0,1,0], "Binaries are incorrect")
        # Roundtrip
        s2 = qubo_tools.x_to_s(x)
        self.assertEqual(np.linalg.norm(s-s2), 0, "Roundtrip spins are not equal")

    def test_evaluators(self):
        """Test objective evaluation"""
        Q = np.arange(0,16).reshape((4,4))
        c = 1.2
        x = np.array([1, 2, 3, 4])
        obj_val = c + Q.dot(x).dot(x)
        obj_val2 = qubo_tools.evaluate_QUBO(Q, c, x)
        self.assertEqual(obj_val, obj_val2, "QUBO objective values are not equal")

        J = np.arange(0,16).reshape((4,4))
        # zero out diagonal
        J[np.arange(4), np.arange(4)] = 0
        h = np.arange(1, 5)
        s = np.arange(-2, 2)
        obj_val = J.dot(s).dot(s) + h.dot(s) + c
        obj_val2 = qubo_tools.evaluate_Ising(J, h, c, s)
        self.assertEqual(obj_val, obj_val2, "Ising objective values are not equal")

    def test_ising_extraction(self):
        """Test getting Ising problem from QUBO"""
        Q = sparse.csr_array(np.arange(0,16).reshape((4,4)))
        d = Q.diagonal()
        J, h = qubo_tools.get_Ising_J_h(Q)
        self.assertEqual(0, np.linalg.norm(d-h), "Diagonals are not equal")
        # Expecting Q to be mutated
        self.assertEqual(0, np.linalg.norm((Q-J).toarray()), "Matrix is not correct")
        self.assertEqual(0, np.linalg.norm(J.diagonal()), "Diagonal is not zero")

    def test_roundtrip_qubo(self):
        """Test roundtrip conversion QUBO -> Ising -> QUBO"""
        Q = np.arange(0,16).reshape((4,4))
        J, h, c = qubo_tools.QUBO_to_Ising(Q)
        Q2, c2 = qubo_tools.Ising_to_QUBO(J, h, c)
        self.assertEqual(0, c2, "QUBO constant has changed")
        self.assertEqual(0, np.linalg.norm(Q-Q2), "QUBO matrix has changed")

    def test_roundtrip_ising(self):
        """Test roundtrip conversion Ising -> QUBO -> Ising"""
        J = np.arange(0,16).reshape((4,4))
        # zero-out diagonal
        J[np.arange(4), np.arange(4)] = 0
        h = np.arange(17, 21)
        Q, c = qubo_tools.Ising_to_QUBO(J, h)
        J2, h2, c2 = qubo_tools.QUBO_to_Ising(Q, c)
        self.assertEqual(0, c2, "Ising constant has changed")
        self.assertEqual(0, np.linalg.norm(h-h2), "Ising vector has changed")
        self.assertEqual(0, np.linalg.norm(J-J2), "Ising matrix has changed")

    def test_pattern_conversion(self):
        """Test conversion to/from symmetric/upper triangular forms"""
        M = np.arange(0,16).reshape((4,4))
        S = qubo_tools.to_symmetric(M)
        S2 = 0.5*(M + M.transpose())
        self.assertEqual(0, np.linalg.norm(S2-S), "Symmetrized matrix not correct")

        U = qubo_tools.to_upper_triangular(M)
        LT = np.tril(M, k=-1)
        U2 = M + LT.transpose() - LT
        self.assertEqual(0, np.linalg.norm(U2-U), "Upper triangular matrix not correct")
        ULT = np.tril(U.toarray(), k=-1)
        self.assertEqual(0, np.linalg.norm(ULT), "Matrix is not upper triangular")

    def test_qubo_container(self):
        """Test QUBOContainer"""
        M = np.arange(0,16).reshape((4,4))
        qubo = qubo_tools.QUBOContainer(M, 0, pattern="upper-triangular")
        U = qubo_tools.to_upper_triangular(M).toarray()
        self.assertEqual(0, np.linalg.norm(U - qubo.Q), "Upper-Tri QUBO matrix is incorrect")
        qubo = qubo_tools.QUBOContainer(M, 0, pattern="symmetric")
        S = qubo_tools.to_symmetric(M).toarray()
        self.assertEqual(0, np.linalg.norm(S - qubo.Q), "Symmetric QUBO matrix is incorrect")
        qubo = qubo_tools.QUBOContainer(M, 0, pattern="foo")
        self.assertEqual(0, np.linalg.norm(M - qubo.Q), "QUBO matrix is incorrect")
