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

    # generate an rainflow matrix with zeros
    xbinsize = matrices[0].xbinsize
    ybinsize = matrices[0].ybinsize
    counts = np.zeros_like(matrices[0].counts)
    output = rfm(xbinsize, ybinsize, counts)

    # summing up the matrix entries
    for mat in matrices:
        output.counts += mat
    return output
