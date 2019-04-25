# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:18:24 2018

@author: smharwo

Test the correctness of the outputted matrices
(QUBO vs Ising)
"""
import numpy as np
import scipy   
import matplotlib.pyplot as plt 

example_name= 'ExMIRPg2'
#example_name= 'ExSmall'


matrixQ = []
with open(example_name+'.qubo') as f:
    numrowsQ = 0
    numcolsQ = 0
    file_lines = f.readlines()
    for line in file_lines:
        if line[0] == 'c':
            # comment line. Get constant of objective if possible;
            # if the line contains an equal sign the constant is after it
            split_contents = line.split(' = ')
            if len(split_contents)>1:
                constantQ = float(split_contents[1])
        elif line[0] == 'p':
            # 'Sentinel' line (specific to Dwave input)
            # p qubo 0 maxDiagonals nDiagonals nElements
            contents = line.split(' ')
            matLength = int(contents[4]) + int(contents[5])
        else:
            # row col val
            contents = line.split(' ')
            row = int(contents[0])
            col = int(contents[1])
            val = float(contents[-1])            
            matrixQ.append((row, col, val))
            numrowsQ = max(numrowsQ,row)
            numcolsQ = max(numcolsQ,col)
assert (len(matrixQ) == matLength), "QUBO matrix length discrepancy"

matrixI = []
with open(example_name+'.rudy') as f:
    numrowsI = 0
    numcolsI = 0
    file_lines = f.readlines()
    for line in file_lines:
        if line[0] == '#':
            # comment line. Get constant of objective if possible;
            # if the line contains an equal sign the constant is after it
            split_contents = line.split(' = ')
            if len(split_contents)>1:
                constantI = float(split_contents[1])
        else:
            contents = line.split(' ')
            if len(contents) == 2:
                # sizing
                # numvars numelements
                matLength = int(contents[1])
            else:
                # row col val
                row = int(contents[0])
                col = int(contents[1])
                val = float(contents[-1])            
                matrixI.append((row, col, val))
                numrowsI = max(numrowsI,row)
                numcolsI = max(numcolsI,col)
assert (len(matrixI) == matLength), "Ising matrix length discrepancy"

data = []
row = []
col = []
for i,j,val in matrixI:
    data.append(val)
    row.append(i)
    col.append(j)
    
denseMI = scipy.sparse.coo_matrix((data,(row, col))).toarray()
plt.spy(denseMI)
plt.show()
       
# we at least expect the size of the matrices to be the same
assert (numrowsQ == numcolsQ), "QUBO matrix not square"
assert (numrowsI == numcolsI), "Ising matrix not square"
assert (numrowsQ == numrowsI), "QUBO and Ising matrices different sizes"

    
numTest = 100
N = numrowsQ + 1 # number of variables = max index  + 1
#np.random.seed(0)

for k in range(numTest):
    # random binary variables and corresponding "spins"
    randomX = np.random.randint(0,2,N)
    randomS = 2*randomX - 1
    # calculate QUBO objective
    objectiveQ = constantQ
    for i,j,val in matrixQ:
        objectiveQ += randomX[i]*val*randomX[j]
    # calculate Ising objective
    objectiveI = constantI
    for i,j,val in matrixI:
        if i == j:
            # linear term
            objectiveI += val*randomS[i]
        else:
            objectiveI += randomS[i]*val*randomS[j]
    assert objectiveQ == objectiveI, "Test FAILED"
    #print(objectiveQ)
    #print(objectiveI)
    
print('Test passed!')