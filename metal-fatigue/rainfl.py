import numpy as np

# definition of a class which represents a rainflow matrix


class rfm:
    def __init__(self, xbinsize, ybinsize, counts):
        self.xbinsize = xbinsize
        self.ybinsize = ybinsize
        # counts should be a numpy array
        self.counts = counts


def add(*matrices):
    """Adds two or more rainflow matrices

    Args:
        *matrices: a list or tuple of Rainflow matrices to add. Must be the same size.
    """
    #check, if matrices are of the same size and shape 
    consistency_check(*matrices)

    # generate an rainflow matrix with zeros
    xbinsize = matrices[0].xbinsize
    ybinsize = matrices[0].ybinsize
    counts = np.zeros_like(matrices[0].counts)
    output = rfm(xbinsize, ybinsize, counts)

    # summing up the matrix entries
    for mat in matrices:
        output.counts += mat
    return output


def consistency_check(*matrices):
    """Compares binsize and shape of the given list of matrices.

    Args:
        *matrices: a list or tuple of Rainflow matrices to add.
    """
    xbinsize = matrices[0].xbinsize
    ybinsize = matrices[0].ybinsize
    shape = matrices[0].counts.shape()
    for mat in matrices:
        if not mat.xbinsize == xbinsize or not mat.ybinsize == ybinsize or not mat.counts.shape() == shape:
            raise ValueError("Rainflow matrices must be of same shape and same size")
    pass
