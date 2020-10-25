import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import xarray as xr

from pycis import mueller_product


class TestMueller(unittest.TestCase):
    def test_mueller_product(self, ):
        """
        basic Mueller matrix multiplication test

        """
        mdims = ('mueller_v', 'mueller_h')
        mm_1 = xr.DataArray(np.random.rand(4, 4, ), dims=mdims, )
        mm_2 = xr.DataArray(np.identity(4, ), dims=mdims, )
        sv_1 = xr.DataArray(np.random.rand(4, ), dims=('stokes', ), )

        assert_almost_equal(mm_1.values, mueller_product(mm_1, mm_2).values, )
        assert_almost_equal(mm_1.values, mueller_product(mm_2, mm_1).values, )
        assert_almost_equal(sv_1.values, mueller_product(mm_2, sv_1).data, )


if __name__ == '__main__':
    unittest.main()
