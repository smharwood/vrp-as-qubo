"""
The purpose of this example is solely to show:

    1) how one can obtain QUBO matrices for a specific example
        NOTE: larger time horizon will result in larger problems,
              different examples (ExSmall, ExMIRPg1) have different properties
              
    2) how through the use of the QUBOContainer class one can access
        both the Ising and the QUBO formulations of the same problem
        
    NOTE: There are a number of other usefull functions in the modules
            QuboContainer.py and TestTools.py that should be working 
            out-of-the-box. Take a look if interested.
"""



import QUBOTools as qt
from arc_based.ExMIRPg2 import getQUBO as getQAB
from sequence_based.ExMIRPg2 import getQUBO as getQSB
from path_based.ExMIRPg2 import getQUBO as getQPB

time_horizon = 16

for getQubo in [getQAB, getQSB, getQPB]:
    print("\n\n*********************************************\n")
    print("Using defintion from ", getQubo.__module__)
    sparseMatrix, constant = getQubo(time_horizon);
    qubo_container = qt.QUBOContainer(sparseMatrix, constant)

    print("---------------------------")
    print("QUBO formulation: Q:{}, const:{}"
            .format(qubo_container.Q, qubo_container.constQ))

    print("---------------------------")
    print("Ising formulation: J:{}, h:{}, const:{}"
            .format(qubo_container.J, qubo_container.h, qubo_container.constI))
            
