import numpy as np
import matplotlib.pyplot as plt
import warnings
import copy


class binned(object):
    def __init__(self, values, binsize, minedge, numbins):
        """Definition of binned object
        Args:
            value (numpy array): numpy array, can be counts (for example in a rfm or a binned time series)
            binsize (float): binsize (same for each dimension)
            minedge (numpy array): minimum value for bin edge of each dimension
            numbins (numpy array): number of bins for each dimension
        Notes:
            Binning convention is: bins[i-1] <= x < bins[i]
        """
        self.binsize = binsize
        self.minedge = minedge
        self.values = values
        self.numbins = numbins
        self.maxedge = minedge + binsize * numbins
        if np.isscalar(binsize):
            self.bins = np.linspace(minedge, minedge + binsize * numbins, numbins + 1)
        else:
            bin_list = []
            for binsize_, minedge_, numbins_ in zip(binsize, minedge, numbins):
                bin_list.append(np.linspace(minedge_, minedge_ + binsize_ * numbins_, numbins_ + 1))
            self.bins = np.asarray(bin_list)

    def multiply_constant(self, constant):
        """Simple multiplication of a rainflow matrix with a given factor

        Args:
            constant (float): floating point/integer factor

        Returns:
            rfm: rainflow matrix object
        """
        return binned(self.values * constant, self.binsize, self.minedge, self.numbins)

    def add_constant(self, constant):
        """Simple multiplication of a rainflow matrix with a given factor

        Args:
            constant (float): floating point/integer factor

        Returns:
            rfm: rainflow matrix object
        """
        return binned(self.values + constant, self.binsize, self.minedge, self.numbins)

    def rebin(self, binsize, xmin, ymin):
        pass


class _rfm(binned):
    def __init__(self, counts, binsize, xmin, ymin, matrixtype):
        """Definition of a class which represents a rainflow matrix

        Args:
            counts (numpy array): numpy array of the counts
            binsize (float): binsize (same for each dimension)
            xmin (float): minimum value of bin edge in 1st dimension
            ymin (float): minimum value of bin edge in 2nd dimension
            matrixtype (string): type of rfm (FromTo/RangeMean)
            bins (numpy array): array of bin edges for each dimension
        Notes:
            Binning convention is: bins[i-1] <= x < bins[i]
        """
        numbins = counts.shape[0]
        binned.__init__(self, values=counts, numbins=np.array([numbins, numbins]), binsize=np.array([binsize, binsize]),
                        minedge=np.array([xmin, ymin]))
        self.xmin = xmin
        self.ymin = ymin
        self.matrixtype = matrixtype

    def plot2d(self, **kwargs):
        """2D Colormap plot of the Rainflow matrix

        Args:
            **kwargs: **kwargs are passed to plt.imshow()

        Returns:
            figure: matplotlib figure object
            axes: matplotlib axes object
        """
        # create fig, ax
        fig = plt.figure()
        ax = fig.add_subplot((111))

        # imshow plot
        cax = ax.imshow(self.values,
                        cmap=plt.get_cmap("Blues"),
                        extent=(self.bins[0][0], self.bins[0][-1], self.bins[1][-1], self.bins[1][0]),
                        **kwargs)

        # create colorbar
        fig.colorbar(cax)

        # create grid and ticks
        ax.grid(which='minor', alpha=0.8, linewidth=0.3)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_minor_locator(plt.FixedLocator(self.bins[0]))
        ax.yaxis.set_minor_locator(plt.FixedLocator(self.bins[0]))
        ax.tick_params(which="minor", direction="in")
        if self.matrixtype == 'FromTo':
            ylabel = 'From'
            xlabel = 'To'
        elif self.matrixtype == 'RangeMean':
            ylabel = 'Range'
            xlabel = 'Mean'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return fig, ax


class from_to(_rfm):
    def __init__(self, counts, binsize, xmin, ymin):
        """Rainflow object of type "FromTo"

        Args:
            counts (numpy array): numpy array of the counts
            binsize (float): binsize (same for each dimension)
            xmin (float): minimum value of bin edge in 1st dimension
            ymin (float): minimum value of bin edge in 2nd dimension
            bins (numpy array): array of bin edges for each dimension

        Notes:
            Binning convention is: bins[i-1] <= x < bins[i]
        """
        _rfm.__init__(self, counts=counts, binsize=binsize, xmin=xmin, ymin=ymin, matrixtype='FromTo')

    def to_range_mean():
        pass


class range_mean(_rfm):
    def __init__(self, counts, binsize, xmin, ymin):
        """Rainflow object of type "RangeMean"

        Args:
            counts (numpy array): numpy array of the counts
            binsize (float): binsize (same for each dimension)
            xmin (float): minimum value of bin edge in 1st dimension
            ymin (float): minimum value of bin edge in 2nd dimension
            bins (numpy array): array of bin edges for each dimension

        Notes:
            Binning convention is: bins[i-1] <= x < bins[i]
        """
        _rfm.__init__(self, counts=counts, binsize=binsize, xmin=xmin, ymin=ymin, matrixtype='RangeMean')

    def to_from_to():
        pass


def zerosrfm_like(matrix):
    """Generates a rainflow matrix filled with zeroes with dimensions like matrix

    Args:
        matrix: a rainflow matrix

    Returns:
        rfm: return rainflow matrix object
    """
    return _rfm(np.zeros_like(matrix.values), matrix.binsize[0], matrix.xmin, matrix.ymin, matrix.matrixtype)


def onesrfm_like(matrix):
    """Generates a rainflow matrix filled with ones with dimensions like matrix

    Args:
        matrix: a rainflow matrix

    Returns:
        rfm: return rainflow matrix object
    """
    return _rfm(np.ones_like(matrix.values), matrix.binsize[0], matrix.xmin, matrix.ymin, matrix.matrixtype)


def addrfm(*matrices):
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
        output.values = output.values + mat.values
    return output


def consistency_check(*matrices):
    """Compares binsize and shape of the given list of rfm.

    Args:
        *matrices: Rainflow matrices to compare.
    """
    binsize = matrices[0].binsize
    numbins = matrices[0].numbins
    minedge = matrices[0].minedge
    matrixtype = matrices[0].matrixtype
    for mat in matrices:
        cbinsize = np.allclose(mat.binsize, binsize)
        cminedge = np.allclose(mat.minedge, minedge)
        cnumbins = np.allclose(mat.numbins, numbins)
        cmattype = matrixtype == mat.matrixtype
        if not (cbinsize and cminedge and cmattype and cnumbins):
            raise ValueError("Rainflow matrices must be of the same shape, type and size")
    pass


def mulitplyrfm(*matrices):
    """Multiplies two or more rainflow matrices

    Args:
        *matrices: Rainflow matrices to multiply. Must be the same size.

    Returns:
        rfm: rainflow matrix object
    """
    # check, if matrices are of the same size and shape
    consistency_check(*matrices)

    # generate an rainflow matrix with ones
    output = onesrfm_like(matrices[0])

    # multiplication of the matrix entries
    for mat in matrices:
        output.values = output.values * mat.values
    return output


def _rainflow_counting_core(point, end, cache):
    """Core of rainflow cycling according to ASTM E1049 − 85 (2017)

    Args:
        point (int): bin number
        end (bool): only True, if last point is reached
        cache (list): residuum of counting

    Returns:
        list: list with from values
        list: list with to values
        list: list with counted cycles
    """
    from_ = []
    to_ = []
    cycles_ = []

    def count_helper(cycles):
        from_.append(Y[0])
        to_.append(Y[1])
        cycles_.append(cycles)
    # step 1
    cache.append(point)
    # step 6
    if end:
        while len(cache) > 1:
            Y = [cache[-2], cache[-1]]
            count_helper(0.5)
            cache.pop()
        return from_, to_, cycles_
    while len(cache) >= 3:
        # step 2
        X = [cache[-2], cache[-1]]
        Y = [cache[-3], cache[-2]]
        # step 3
        if np.abs(X[0] - X[1]) < np.abs(Y[0] - Y[1]):
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
        cache.pop(0)
    return from_, to_, cycles_


def binned_rainflow(binned_turningpoints, matrix=None, cache=None):
    """Rainflow cycle counting  with binned data

    Args:
        binned_turningpoints (binned object): turning points (binned) of the signal
        matrx (rfm object, optional): FromTo rfm object to write on. Bins must be conform with binned_turningpoints. None will create an empty matrix
        cache (list, optional): cache from last counting iteration

    Returns:
        rfm object: FromTo rainflow matrix object
        list: cache (residuum) from counting
    """
    if matrix is None:
        numbins = binned_turningpoints.numbins
        binsize = binned_turningpoints.binsize
        minedge = binned_turningpoints.minedge
        zeros = np.zeros((numbins, numbins))
        matrix = from_to(zeros, binsize, minedge, minedge)
    if cache is None:
        cache = []
    end = False
    for i, point in enumerate(binned_turningpoints.values):
        if i == (len(binned_turningpoints.values) - 1):
            end = True
        from_, to_, cycles_ = _rainflow_counting_core(point, end, cache)
        for f_, t_, c_ in zip(from_, to_, cycles_):
            matrix.values[f_, t_] = matrix.values[f_, t_] + c_
    return matrix, cache


def continuous_rainflow(turning_points, cache=None):
    """Rainflow counting without binning according to according to ASTM E1049 − 85 (2017)

    Args:
        turning_points (numpy array): turning points of time series
        cache (list, optional): cache from last counting iteration

    Returns:
        numpy array: array with range values
        numpy array: array with mean values
        numpy array: array with counted cycles
    """
    if cache is None:
        cache = []
    range_ = []
    mean_ = []
    cycle = []
    end = False
    for i, point in enumerate(turning_points):
        if i == (len(turning_points) - 1):
            end = True
        from_, to_, cycles_ = _rainflow_counting_core(point, end, cache)
        for f_, t_, c_ in zip(from_, to_, cycles_):
            range_.append(np.abs(f_ - t_))
            mean_.append((f_ + t_) / 2.)
            cycle.append(c_)
    return np.array(range_), np.array(mean_), np.array(cycle), cache


def rainflow(series, numbins=128, minedge=None, maxedge=None):
    """Rainflow cycle counting with binning

    Args:
        series (numpy array): series for counting
        numbins (int, optional): number of bins
        minedge (float, optional): minimum value of bin edge 
        maxedge (float, optional): maximum value of bin edge

    Returns:
        rfm object: FromTo rainflow matrix object
        list: cache (residuum) from counting
    """
    if minedge is None:
        minedge = np.min(series)
    if maxedge is None:
        maxedge = np.max(series) * 1.01
    binned_series = bin_series(series, minedge, maxedge, numbins)
    turn_p = copy.deepcopy(binned_series)
    turn_p.values = binned_series.values[turning_points(binned_series.values)]
    rfm, cache = binned_rainflow(turn_p)
    return rfm, cache


def bin_series(series, minedge, maxedge, numbins):
    """Digitize (bin) data

    Args:
        series (numpy array): numpy array to digitize
        minedge (float): minimum value of bin edge
        maxedge (float): maximum value of bin edge
        numbins (int): number of bins

    Returns:
        binned object: binned data
    """
    # warning, if overflow
    if minedge > np.min(series) or maxedge <= np.max(series):
        warnings.warn("Matrix overflow. Check min and max values.")

    # series to turnuíng points
    bins = np.linspace(minedge, maxedge, numbins + 1)
    dig = np.digitize(series, bins) - 1
    binsize = bins[1] - bins[0]
    hist = binned(dig, binsize, minedge, numbins)
    return hist


def turning_points(series):
    """Estimating the index values of the turning points in a series

    Args:
        series (numpy array): series for estimating turning points

    Returns:
        numpy array: index array
    """
    cache = []
    index = []
    for i, point in enumerate(series):
        if i == 0:
            continue
        if i == (len(series) - 1):
            if series[index[0]] != series[0]:
                index = np.insert(index, 0, 0)
            if series[index[-1]] != series[-1]:
                index = np.insert(index, len(index), len(series) - 1)
            break
        nex = series[i + 1]
        prev = series[i - 1]
        cache.append(point)
        if ((point > nex) and (point < prev)) or ((point < nex) and (point > prev)):
            cache.pop()
        elif ((point < nex) and (point < prev)) or ((point > nex) and (point > prev)):
            index.append(i)
            cache = [point]
        elif np.all(point >= cache[0]) and (point > nex):
            index.append(i)
            cache = [point]
        elif np.all(point <= cache[0]) and (point < nex):
            index.append(i)
            cache = [point]
        else:
            cache.pop()
    return np.array(index)
