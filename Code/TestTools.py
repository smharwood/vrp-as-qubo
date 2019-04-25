# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:18:24 2018

@author: smharwo

Some tools to assess Ising matrices
"""
import sys, os.path
import numpy as np
import scipy.sparse 
import matplotlib.pyplot as plt 


def loadSpins(filename):
    """
    Read spins saved in textfile
    Either whitespace-separated on one line,
    or one entry per line
    """
    with open(filename) as f:
        lines = f.readlines()
        if len(lines) == 1:
            spins = [int(s) for s in lines[0].split()]
            return spins
        else:
            spins = [int(s) for s in lines]
            return spins
        
    
def loadMatrix(filename, comment_char):
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
    with open(filename) as f:
        numrows = 0
        numcols = 0
        file_lines = f.readlines()
        for line in file_lines:
            if line[0] == comment_char:
                # comment line. Get constant of objective if possible;
                # if the line contains an equal sign the constant is after it
                split_contents = line.split('=')
                if len(split_contents)>1:
                    constant = float(split_contents[1])
            elif line[0] == 'p':
                # 'Sentinel' line (specific to Dwave input- indicates QUBO)
                # p qubo 0 maxDiagonals nDiagonals nElements
                contents = line.split()
                matLength = int(contents[4]) + int(contents[5])
            else:
                # "split()" splits based on arbitrary whitespace
                contents = line.split()
                if len(contents) == 2:
                    # sizing for ISING form
                    # numvars numelements
                    matLength = int(contents[1])
                else:
                    # row col val
                    row.append(int(contents[0]))
                    col.append(int(contents[1]))
                    data.append(float(contents[2]))        
                    numrows = max(numrows,max(row))
                    numcols = max(numcols,max(col))
    # end with

    # A few sanity checks
    assert (len(row) == matLength), "Input matrix length discrepancy"
    assert (numrows == numcols), "Input matrix not square"

    sparseMatrix = scipy.sparse.coo_matrix((data,(row, col)))
    return sparseMatrix, constant

    
# Helpers to help catch wrong inputs
def loadQUBOMatrix(filename):
    return loadMatrix(filename, 'c')

def loadIsingMatrix(filename):
    return loadMatrix(filename, '#')


def evaluateIsingObjective(matrix, constant, spins):
    """
    Given a vector of {-1,+1} spins/variables,
    evaluate the objective/energy of the system:
    constant + spins^T (Matrix diagonal) + spins^T (Matrix without diagonal)*spins 
    """
    (r,c) = matrix.shape
    assert (r == c), "Matrix not square"
    assert (r == len(spins)), "Number of spins incorrect for matrix"
    
    # calculate Ising objective
    diag = matrix.diagonal()
    objective = constant + diag.dot(spins) + matrix.dot(spins).dot(spins) - sum(diag)

    # Subtracting out sum of diagonal is equivalent and is faster than making a copy:
    #MminusDiag = matrix.copy()
    #MminusDiag.setdiag(0)
    #objective = constant + diag.dot(spins) + MminusDiag.dot(spins).dot(spins)

    return objective


def exhaustiveSearch(matrix, constant, stopAtFeasible=False):
    """
    Go thru all possible spins to find min
    """
    numSpins = matrix.shape[0]
    
    bestObj = np.inf
    bestSpins = []
    progCount = 0
    progSpacing = (2**numSpins)/10000.0
    for v in range(2**numSpins):
        # get binary representation of integer with leading zeros
        # then convert to {-1, +1} -values from {0,1}-valued
        spins = [(2*int(s) - 1) for s in format(v, '0{}b'.format(numSpins))]
        
        if v > progCount*progSpacing:
            progCount = int(np.floor(v/progSpacing)+1)
            print('\rConfigurations evaluated: {}%'.format(progCount/100.0), end='')
            
        obj = evaluateIsingObjective(matrix, constant, spins)
        if obj < bestObj:
            bestObj = obj
            bestSpins = [spins]
        elif obj == bestObj:
            # if we have a tie, append the spins
            bestSpins.append(spins)
            
        if stopAtFeasible and bestObj == 0:
            break
        
    return bestObj, bestSpins
    

def visualizeIsingMatrixSparsity(matrix):
    """
    What does the matrix look like?
    """
    DenseMatrix = matrix.toarray()
    plt.spy(DenseMatrix)
    plt.show()
    
    
def compareQuboAndIsing(exName):
    """
    Given a root example name exName, load the data in
    exName.qubo and exName.rudy
    and compare the objective functions on equivalent spins
    to make sure that the conversion from QUBO to Ising is corect
    """
    QM, QC = loadQUBOMatrix(exName+'.qubo1')
    IM, IC = loadIsingMatrix(exName+'.rudy1')
    
    # we at least expect the size of the matrices to be the same
    assert (QM.shape[0] == QM.shape[1]), "QUBO matrix not square"
    assert (IM.shape[0] == IM.shape[1]), "Ising matrix not square"
    assert (QM.shape == IM.shape), "QUBO and Ising matrices different sizes"
    
    numTest = 100   # number of tests to perform
    N = QM.shape[0] # number of variables
    
    for k in range(numTest):
        # random binary variables and corresponding "spins"
        randomX = np.random.randint(0,2,N)
        randomS = 2*randomX - 1
        # calculate QUBO objective
        objectiveQ = QC + QM.dot(randomX).dot(randomX)
        # calculate Ising objective
        objectiveI = evaluateIsingObjective(IM, IC, randomS)
        assert objectiveQ == objectiveI, "Test FAILED"
        
    print('Test passed!')
    

if __name__ == "__main__":
    
    args = sys.argv[1:]
    assert len(args) >= 1, "Need at least a matrix filename"
        
    # If we accidentally put the file names in quotes strip those out
    matrix_filename = args[0].strip('\'\"')
    print("\nUsing matrix from "+matrix_filename)
    assert (os.path.isfile(matrix_filename)), "Matrix file "+matrix_filename+" not found"

    matrix,constant = loadIsingMatrix(matrix_filename)
    print("Matrix sparsity:")
    visualizeIsingMatrixSparsity(matrix)
    
    if len(args) == 1:
        print("\nPerforming exhaustive search for minimum")
        bestObj, bestSpin = exhaustiveSearch(matrix,constant)
        print("\nBest objective = {} at {}".format(bestObj, bestSpin))
    else :
        # If we accidentally put the file names in quotes strip those out
        spins_filename = args[1].strip('\'\"')
        print("\nEvaluating spins from "+spins_filename)
        assert (os.path.isfile(spins_filename)), "Spins file "+spins_filename+" not found"
    
        spins = loadSpins(spins_filename)
        print("Spins: "+str(spins))
        
        obj = evaluateIsingObjective(matrix, constant, spins)
        print("\nObjective is {}".format(obj))

