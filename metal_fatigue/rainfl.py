import numpy as np

# definition of a class which represents a rainflow matrix


class rfm(object):
    def __init__(self, counts, binsize, xmin, ymin):
        self.binsize = binsize
        self.xmin = xmin
        self.ymin = ymin
        # counts should be a numpy array
        self.counts = counts


def zerosrfm_like(matrix):
    """Generates a rainflow matrix filled with zeroes only on the same scale as matrix

    Args:
        matrix: a rainflow matrix

    Returns:
        rfm: return rainflow matrix object
    """
    return rfm(np.zeros_like(matrix.counts), matrix.binsize, matrix.xmin, matrix.ymin)


def onesrfm_like(matrix):
    """Generates a rainflow matrix filled with ones only on the same scale as matrix

    Args:
        matrix: a rainflow matrix

    Returns:
        rfm: return rainflow matrix object
    """
    return rfm(np.ones_like(matrix.counts), matrix.binsize, matrix.xmin, matrix.ymin)


def add(*matrices):
    """Adds two or more rainflow matrices

    Args:
        *matrices: Rainflow matrices to add. Must be the same size.

    Returns:
        rfm: return rainflow matrix object
    """
    # check, if matrices are of the same size and shape
    consistency_check(*matrices)

    # generate an rainflow matrix with zeros
    output = zerosrfm_like(matrices[0])

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
    xmin = matrices[0].xmin
    ymin = matrices[0].ymin
    for mat in matrices:
        if not (mat.binsize == binsize and mat.counts.shape == shape and mat.xmin == xmin and mat.ymin == ymin):
            raise ValueError("Rainflow matrices must be of same shape and same size")
    pass


def mulitply(*matrices):
    """Multiplies two or more rainflow matrices

    Args:
        *matrices: Rainflow matrices to multiply. Must be the same size.

    Returns:
        rfm: return rainflow matrix object
    """
    # check, if matrices are of the same size and shape
    consistency_check(*matrices)

    # generate an rainflow matrix with ones
    output = onesrfm_like(matrices[0])

    # multiplication of the matrix entries
    for mat in matrices:
        output.counts = output.counts * mat.counts
    return output


def extrapolate(matrix, factor):
    """Simple extrapolation of a rainflow matrix by a given factor

    Args:
        matrix (rfm): Rainflow matrix object
        factor (float): floating point/integer factor

    Returns:
        rfm: rainflow matrix object
    """
    return rfm(matrix.counts * factor, matrix.binsize, matrix.xmin, matrix.ymin)


def rainflow_count(series, min, max, binsize):
    """Performs rainflow cycle counting and digitizing on a turning point series. Counting occurs according to ASTM E1049 − 85 (2017).
    (Adds a bin if (max-min)/binsize is not an integer)


    Args:
        series (numpy array): turning points
        min (float): minimum value of bin
        max (float): maximum value of bin
        binsize (float): size of bins

    Returns:
        rfm: rainflow matrix
    """
    # series to turnuíng points
    bins = np.range(min, max, binsize)  # round up
    turning_points = np.digitize(seris, np.ceil(bins))
    size = np.size(bins)

    # init empty matrix
    zeroes = np.zeroes((size, size))
    output = rfm(zeroes, binsize, min, min)

    cache = []

    def count_helper(cycles):
        if Y > 0:
            i = cache[-3]
            j = cache[-2]
        else:
            i = cache[-2]
            j = cache[-3]
        output.counts[i, j] = output.counts[i, j] + cycles

    for i, point in enumerate(turning_points):
        # step 1
        cache.append(point)

        # step 6
        if i == np.size(turning_points):
            while len(cache) > 1:
                Y = cache[-2] - cache[-1]
                count_helper(0.5)
                cache.pop()
            continue

        while len(cache) >= 3:
            # step 2
            X = cache[-2] - cache[-1]
            Y = cache[-3] - cache[-2]

            # step 3
            if np.abs(X) < np.abs(Y):
                break
            # step 4
            elif len(cache) > 3:
                count_helper(1)
                last = cache.pop()
                cache.pop()
                cache.pop()
                cache.append(last)
                continue
            # step 5
            count_helper(0.5)
            cache.reverse()
            cache.pop()
            cache.reverse()
    return output
