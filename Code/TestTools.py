# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:18:24 2018

@author: smharwo

Some tools to assess Ising matrices
"""
import os.path, argparse
import numpy as np
import scipy.sparse 
# import matplotlib.pyplot as plt 


def main():
    parser = argparse.ArgumentParser(description=
            "Tools to assess Ising problems for routing problems.\n"+
            "Can evaluate given spins or search for spins.\n"+
            "Assuming structure of TestSet problems, can assess feasibility of spins",
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i','--input',type=str,
                        help="Name of file containing Ising matrix in sparse format")
    parser.add_argument('-e','--eval',type=str,action='store',
                        help="Name of file containing spins to evaluate")
    parser.add_argument('-s','--search',action='store_const',const=True,
                        help="Perform exhaustive search for minimizing spins")
    args = parser.parse_args()
    
    # if no input file set, print help
    if not args.input:
        parser.print_help()
        return
    
    # load matrix
    matrix_filename = args.input
    print("\nUsing matrix from "+matrix_filename)
    assert os.path.isfile(matrix_filename), "Matrix file "+matrix_filename+" not found"
    matrix,constant = loadIsingMatrix(matrix_filename)
    print("\nConstant term to add to objective: {}".format(constant))
    #print("Matrix sparsity:")
    #visualizeIsingMatrixSparsity(matrix)
    
        
    if args.eval:
        spins_filename = args.eval
        print("\nEvaluating spins from "+spins_filename)
        assert os.path.isfile(spins_filename), "Spins file "+spins_filename+" not found"
        spins = loadSpins(spins_filename)
        print("Spins: "+str(spins))
        
        obj = evaluateIsingObjective(matrix, constant, spins)
        print("\nObjective is {}".format(obj))
        
        # Check for a feasibility problem definition;
        # This assumes the specific organization of the test set
        fname_split = matrix_filename.split('.')
        m_fname_root = fname_split[0]
        extension = fname_split[1]
        if m_fname_root[-1] == 'f':
            # Already a feasibility problem
            if obj == 0.0:
                print("Spins are feasible")
            else:
                print("Spins are INfeasible, violation = {}".format(obj))
        else:
            feas_matrix_filename = m_fname_root[0:-1] + 'f.'+extension
            if os.path.isfile(feas_matrix_filename):
                print("Corresponding feasibility problem definition "+feas_matrix_filename+" found")
                feas_matrix,feas_constant = loadIsingMatrix(feas_matrix_filename)
                feas_obj = evaluateIsingObjective(feas_matrix, feas_constant, spins)
                if feas_obj == 0.0:
                    print("Spins are feasible")
                else:
                    print("Spins are INfeasible, violation = {}".format(feas_obj))
            else:
                print("Cannot assess feasibility of spins")

    if args.search:
        print("\nPerforming exhaustive search for minimum")
        bestObj, bestSpin = exhaustiveSearch(matrix,constant)
        print("\nBest objective = {} at {}".format(bestObj, bestSpin))

        
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

def evaluateQUBOObjective(matrix, constant, x):
    """ Evaluate QUBO objective: x^T M x + constant """
    (r, c) = matrix.shape
    assert (r == c), "Matrix not square"
    assert (r == len(x)), "Number of spins incorrect for matrix"

    return matrix.dot(x).dot(x) + constant


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
    

# def visualizeIsingMatrixSparsity(matrix):
#     """
#     What does the matrix look like?
#     """
#     DenseMatrix = matrix.toarray()
#     plt.spy(DenseMatrix)
#     plt.show()
    
    
def compareQuboAndIsing(exName):
    """
    Given a root example name exName, load the data in
    exName.qubo and exName.rudy
    and compare the objective functions on equivalent spins
    to make sure that the conversion from QUBO to Ising is correct
    """
    QM, QC = loadQUBOMatrix(exName+'.qubo')
    IM, IC = loadIsingMatrix(exName+'.rudy')
    
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
    main()