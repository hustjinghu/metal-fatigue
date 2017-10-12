import numpy as np

# definition of a class which represents a rainflow matrix


class rfm(object):
    def __init__(self, binsize, counts):
        self.binsize = binsize
        # counts should be a numpy array
        self.counts = counts


def add(*matrices):
    """Adds two or more rainflow matrices

    Args:
        *matrices: Rainflow matrices to add. Must be the same size.
    """
    # check, if matrices are of the same size and shape
    consistency_check(*matrices)

    # generate an rainflow matrix with zeros
    binsize = matrices[0].binsize
    counts = np.zeros_like(matrices[0].counts)
    output = rfm(binsize, counts)

    # summing up the matrix entries
    for mat in matrices:
        output.counts = output.counts + mat.counts
    return output


def consistency_check(*matrices):
    """Compares binsize and shape of the given list of matrices.

    Args:
        *matrices: Rainflow matrices to compare.
    """
    binsize = matrices[0].binsize
    shape = matrices[0].counts.shape
    for mat in matrices:
        if not mat.binsize == binsize or not mat.counts.shape == shape:
            raise ValueError("Rainflow matrices must be of same shape and same size")
    pass


def mulitply(*matrices):
    """Multiplies two or more rainflow matrices

    Args:
        *matrices: Rainflow matrices to multiply. Must be the same size.
    """
    # check, if matrices are of the same size and shape
    consistency_check(*matrices)

    # generate an rainflow matrix with ones
    binsize = matrices[0].binsize
    counts = np.ones_like(matrices[0].counts)
    output = rfm(binsize, counts)

    # summing up the matrix entries
    for mat in matrices:
        output.counts = output.counts * mat.counts
    return output
