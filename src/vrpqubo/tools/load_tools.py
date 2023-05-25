"""
20 December 2018
SM Harwood

Some tools to assess Ising matrices
Note that these methods prefer to work with sparse matrices
"""
import numpy as np
import scipy.sparse

def load_spins(filename):
    """
    Read spins saved in textfile
    Assumes values are separated by whitespace and line breaks,
    reading left to right and top to bottom.
    """
    with open(filename, encoding="utf-8") as file:
        lines = file.readlines()
        spins = np.array([int(float(s)) for line in lines for s in line.split()], dtype=np.short)
        return spins

def load_matrix(filename, comment_char):
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
    mat_length = None
    with open(filename, encoding="utf-8") as f:
        numrows = 0
        numcols = 0
        file_lines = f.readlines()
        for line in file_lines:
            if line[0] == comment_char:
                # comment line. Get constant of objective if possible;
                # if the line contains an equal sign the constant is after it
                split_contents = line.split('=')
                if len(split_contents) > 1:
                    constant = float(split_contents[1])
            elif line[0] == 'p':
                # 'Sentinel' line (specific to Dwave input- indicates QUBO)
                # p qubo 0 maxDiagonals nDiagonals nElements
                contents = line.split()
                mat_length = int(contents[4]) + int(contents[5])
            else:
                # "split()" splits based on arbitrary whitespace
                contents = line.split()
                if len(contents) == 2:
                    # Potentially get sizing for ISING form
                    # numvars numelements
                    mat_length = int(contents[1])
                else:
                    # row col val
                    row.append(int(contents[0]))
                    col.append(int(contents[1]))
                    data.append(float(contents[2]))
                    numrows = max(numrows,max(row))
                    numcols = max(numcols,max(col))
    # end with
    # A few sanity checks
    assert (numrows == numcols), "Input matrix not square"
    if mat_length is not None:
        assert (len(row) == mat_length), "Input matrix length discrepancy"
    # Construct and return
    sparse_matrix = scipy.sparse.coo_array((data,(row, col)))
    return sparse_matrix, constant

def load_qubo_matrix(filename):
    """ Helper for loading QUBOs """
    return load_matrix(filename, 'c')

def load_ising_matrix(filename):
    """ Helper for loading Ising problems """
    return load_matrix(filename, '#')
