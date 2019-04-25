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
        
        
def loadIsingMatrix(filename):
    """
    Load matrix defining the Ising form of problem
    """
    # Store sparsely
    data = []
    row = []
    col = []
    constant = 0
    with open(filename) as f:
        numrowsI = 0
        numcolsI = 0
        file_lines = f.readlines()
        for line in file_lines:
            if line[0] == '#':
                # comment line. Get constant of objective if possible;
                # if the line contains an equal sign the constant is after it
                split_contents = line.split('=')
                if len(split_contents)>1:
                    constant = float(split_contents[1])
            else:
                # "split()" splits based on arbitrary whitespace
                contents = line.split()
                if len(contents) == 2:
                    # sizing
                    # numvars numelements
                    matLength = int(contents[1])
                else:
                    # row col val
                    row.append(int(contents[0]))
                    col.append(int(contents[1]))
                    data.append(float(contents[2]))        
                    numrowsI = max(numrowsI,max(row))
                    numcolsI = max(numcolsI,max(col))
    # end with

    # A few sanity checks
    assert (len(row) == matLength), "Ising matrix length discrepancy"
    assert (numrowsI == numcolsI), "Ising matrix not square"

    sparseIsingMatrix = scipy.sparse.coo_matrix((data,(row, col)))
    
    return sparseIsingMatrix, constant


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


def exhaustiveSearch(matrix, constant, stopAtFeas=False):
    """
    Go thru all possible spins to find min
    """
    numSpins = matrix.shape[0]
    
    bestObj = np.inf
    progCount = 0
    progSpacing = (2**numSpins)/10000.0
    for v in range(2**numSpins):
        # get binary representation of integer with leading zeros
        # then convert to {-1, +1} -values from {0,1}-valued
        spins = [(2*int(s) - 1) for s in format(v, '0{}b'.format(numSpins))]
        
        if v > progCount*progSpacing:
            progCount = int(np.floor(v/progSpacing)+1)
            print(progCount/100.0, "% configurations evaluated.", end = '\r')
            
        obj = evaluateIsingObjective(matrix, constant, spins)
        if obj < bestObj:
            bestObj = obj
            bestSpin = spins
            
        if stopAtFeas and bestObj == 0:
            break
        
    return bestObj, bestSpin
    

def visualizeIsingMatrixSparsity(matrix):
    """
    What does the matrix look like?
    """
    DenseMatrix = matrix.toarray()
    plt.spy(DenseMatrix)
    plt.show()
    

if __name__ == "__main__":
    
    assert len(sys.argv) >= 2, "Need at least a matrix filename"
        
    # If we accidentally put the file names in quotes strip those out
    matrix_filename = sys.argv[1].strip('\'\"')
    print("\nUsing matrix from "+matrix_filename)
    assert (os.path.isfile(matrix_filename)), "Matrix file "+matrix_filename+" not found"

    matrix,constant = loadIsingMatrix(matrix_filename)
    print("Matrix sparsity:")
    visualizeIsingMatrixSparsity(matrix)
    
    if len(sys.argv) == 2:
        print("\nPerforming exhaustive search for minimum")
        bestObj, bestSpin = exhaustiveSearch(matrix,constant)
        print("\nBest objective = {} at {}".format(bestObj, bestSpin))
    else :
        # If we accidentally put the file names in quotes strip those out
        spins_filename = sys.argv[2].strip('\'\"')
        print("\nEvaluating spins from "+spins_filename)
        assert (os.path.isfile(spins_filename)), "Spins file "+spins_filename+" not found"
    
        spins = loadSpins(spins_filename)
        print("Spins: "+str(spins))
        
        obj = evaluateIsingObjective(matrix, constant, spins)
        print("\nObjective is {}".format(obj))
