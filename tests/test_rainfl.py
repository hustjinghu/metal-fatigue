import pytest
import sys
sys.path.append("D:\\alexa\\Documents\\metal-fatigue")
import numpy as np
from metal_fatigue import rainfl


def test_binned():
    # init bin
    values = np.array([[1, 2, 3], [4, 5, 6]])
    binsize = np.array([0.5, 10])
    minedge = np.array([-1.19, 20])
    numbins = np.array([2, 3])
    bin1 = np.array([-1.19, -0.69, -0.19])
    bin2 = np.array([20., 30., 40., 50.])
    bins = np.array((bin1, bin2))
    maxedge = np.array([-0.19, 50])
    binned_object = rainfl.binned(values, binsize, minedge, numbins)
    np.testing.assert_allclose(values, binned_object.values)
    np.testing.assert_allclose(binsize, binned_object.binsize)
    np.testing.assert_allclose(minedge, binned_object.minedge)
    np.testing.assert_allclose(numbins, binned_object.numbins)
    np.testing.assert_allclose(maxedge, binned_object.maxedge)
    for bins_, mat_bins in zip(bins, binned_object.bins):
        np.testing.assert_allclose(bins_, mat_bins)

    # multiply_constant
    result = values * 5.1
    scaled = binned_object.multiply_constant(5.1)
    np.testing.assert_allclose(result, scaled.values)

    # add_constant
    result = values + 5.1
    added = binned_object.add_constant(5.1)
    np.testing.assert_allclose(result, added.values)


def test_rfm():
    values = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, -3]])
    binsize = 1.4
    xmin = 1
    ymin = -5.2
    matrixtype = "FromTo"
    matrix = rainfl._rfm(values, binsize, xmin, ymin, matrixtype)
    np.testing.assert_allclose(values, matrix.values)
    np.testing.assert_allclose(binsize, matrix.binsize[0])
    np.testing.assert_allclose(binsize, matrix.binsize[1])
    np.testing.assert_allclose(xmin, matrix.xmin)
    np.testing.assert_allclose(ymin, matrix.ymin)
    np.testing.assert_allclose(xmin, matrix.minedge[0])
    np.testing.assert_allclose(ymin, matrix.minedge[1])
    matrix.plot2d()
    assert matrixtype == matrix.matrixtype


def test_from_to():
    values = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, -3]])
    binsize = 1.4
    xmin = 1
    ymin = -5.2
    numbins = 3
    matrix = rainfl.from_to(values, binsize, xmin, ymin)
    np.testing.assert_allclose(values, matrix.values)
    np.testing.assert_allclose(binsize, matrix.binsize[0])
    np.testing.assert_allclose(binsize, matrix.binsize[1])
    np.testing.assert_allclose(xmin, matrix.xmin)
    np.testing.assert_allclose(ymin, matrix.ymin)
    np.testing.assert_allclose(xmin, matrix.minedge[0])
    np.testing.assert_allclose(ymin, matrix.minedge[1])
    np.testing.assert_allclose(numbins, matrix.numbins[0])
    np.testing.assert_allclose(numbins, matrix.numbins[1])
    assert "FromTo" == matrix.matrixtype


def test_range_mean():
    values = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, -3]])
    binsize = 1.4
    xmin = 1
    ymin = -5.2
    matrix = rainfl.range_mean(values, binsize, xmin, ymin)
    np.testing.assert_allclose(values, matrix.values)
    np.testing.assert_allclose(binsize, matrix.binsize[0])
    np.testing.assert_allclose(binsize, matrix.binsize[1])
    np.testing.assert_allclose(xmin, matrix.xmin)
    np.testing.assert_allclose(ymin, matrix.ymin)
    np.testing.assert_allclose(xmin, matrix.minedge[0])
    np.testing.assert_allclose(ymin, matrix.minedge[1])
    assert "RangeMean" == matrix.matrixtype


def test_zerosrfm_like():
    values = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, -3]])
    binsize = 1.4
    xmin = 1
    ymin = -5.2
    matrixtype = "FromTo"
    matrix = rainfl._rfm(values, binsize, xmin, ymin, matrixtype)
    zero_matrix = rainfl.zerosrfm_like(matrix)
    np.testing.assert_allclose(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), zero_matrix.values)
    np.testing.assert_allclose(binsize, zero_matrix.binsize[0])
    np.testing.assert_allclose(binsize, zero_matrix.binsize[1])
    np.testing.assert_allclose(xmin, zero_matrix.xmin)
    np.testing.assert_allclose(ymin, zero_matrix.ymin)
    np.testing.assert_allclose(xmin, zero_matrix.minedge[0])
    np.testing.assert_allclose(ymin, zero_matrix.minedge[1])
    assert "FromTo" == zero_matrix.matrixtype


def test_onesrfm_like():
    values = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, -3]])
    binsize = 1.4
    xmin = 1
    ymin = -5.2
    matrixtype = "FromTo"
    matrix = rainfl._rfm(values, binsize, xmin, ymin, matrixtype)
    zero_matrix = rainfl.zerosrfm_like(matrix)
    np.testing.assert_allclose(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), zero_matrix.values)
    np.testing.assert_allclose(binsize, zero_matrix.binsize[0])
    np.testing.assert_allclose(binsize, zero_matrix.binsize[1])
    np.testing.assert_allclose(xmin, zero_matrix.xmin)
    np.testing.assert_allclose(ymin, zero_matrix.ymin)
    np.testing.assert_allclose(xmin, zero_matrix.minedge[0])
    np.testing.assert_allclose(ymin, zero_matrix.minedge[1])
    assert "FromTo" == zero_matrix.matrixtype


def test_addrfm():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    rfm1 = rainfl.from_to(a, 0.1, -1, 1)
    rfm2 = rainfl.from_to(b, 0.1, -1, 1)
    result = rainfl.addrfm(rfm1, rfm2)
    np.testing.assert_allclose(a + b, result.values)
    np.testing.assert_allclose(0.1, result.binsize[0])
    np.testing.assert_allclose(-1, result.xmin)
    np.testing.assert_allclose(1, result.ymin)


def test_multiplyrfm():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    rfm1 = rainfl.from_to(a, 0.1, -1, 1)
    rfm2 = rainfl.from_to(b, 0.1, -1, 1)
    result = rainfl.mulitplyrfm(rfm1, rfm2)
    np.testing.assert_allclose(a * b, result.values)
    np.testing.assert_allclose(0.1, result.binsize[0])
    np.testing.assert_allclose(-1, result.xmin)
    np.testing.assert_allclose(1, result.ymin)


def test_consistency_check():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    c = np.random.rand(4, 4)
    rfm1 = rainfl.from_to(a, 0.1, -1, 1)
    rfm2 = rainfl.from_to(b, 0.1, -1, 2)
    rfm3 = rainfl.from_to(c, 0.1, -1, 1)
    with pytest.raises(ValueError):
        rainfl.consistency_check(rfm1, rfm2, rfm3)


def test_continuous_rainflow():
    series = np.array([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    meanvals = np.array([-0.5, -1., 1., 1., 1., 0., 0.5])
    rangevals = np.array([3, 4, 4, 8, 6, 8, 9])
    cyclevals = np.array([0.5, 0.5, 1., 0.5, 0.5, 0.5, 0.5])
    res = [5]
    ra, mea, cyc, re = rainfl.continuous_rainflow(series)
    np.testing.assert_allclose(rangevals, ra)
    np.testing.assert_allclose(meanvals, mea)
    np.testing.assert_allclose(cyclevals, cyc)
    np.testing.assert_allclose(res, re)


if __name__ == '__main__':
    pytest.main()
