import pytest
import sys
sys.path.append("D:\\alexa\\Documents\\metal-fatigue")
import numpy as np
from metal_fatigue import rainfl


def test_rfm():
    a = np.array([[1, 3e6], [2., -2]])
    b = np.array([[1, 3e6], [2., -2]])
    matrix = rainfl._rfm(a, 1.3, 2, -3, 'FromTo')
    np.testing.assert_allclose(b, matrix.counts)
    np.testing.assert_allclose(1.3, matrix.binsize)
    np.testing.assert_allclose(2, matrix.xmin)
    np.testing.assert_allclose(-3, matrix.ymin)
    np.testing.assert_allclose(b * 3, matrix.extrapolate(3).counts)
    assert 'FromTo' == matrix.mattype


def test_add():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    rfm1 = rainfl.from_to(a, 0.1, -1, 1)
    rfm2 = rainfl.from_to(b, 0.1, -1, 1)
    result = rainfl.add(rfm1, rfm2)
    np.testing.assert_allclose(a + b, result.counts)
    np.testing.assert_allclose(0.1, result.binsize)
    np.testing.assert_allclose(-1, result.xmin)
    np.testing.assert_allclose(1, result.ymin)


def test_multiply():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    rfm1 = rainfl.range_mean(a, 0.1, -1, 1)
    rfm2 = rainfl.range_mean(b, 0.1, -1, 1)
    result = rainfl.mulitply(rfm1, rfm2)
    np.testing.assert_allclose(a * b, result.counts)
    np.testing.assert_allclose(0.1, result.binsize)
    np.testing.assert_allclose(-1, result.xmin)
    np.testing.assert_allclose(1, result.ymin)


if __name__ == '__main__':
    pytest.main()
