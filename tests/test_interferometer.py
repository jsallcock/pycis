import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import xarray as xr
from pycis import mueller_product, UniaxialCrystal, Waveplate


class TestMueller(unittest.TestCase):
    def test_mueller_product(self, ):
        """
        Basic Mueller matrix multiplication test.
        """
        mdims = ('mueller_v', 'mueller_h')
        mm_1 = xr.DataArray(np.random.rand(4, 4, ), dims=mdims, )
        mm_2 = xr.DataArray(np.identity(4, ), dims=mdims, )
        sv_1 = xr.DataArray(np.random.rand(4, ), dims=('stokes', ), )

        assert_almost_equal(mm_1.values, mueller_product(mm_1, mm_2).values, )
        assert_almost_equal(mm_1.values, mueller_product(mm_2, mm_1).values, )
        assert_almost_equal(sv_1.values, mueller_product(mm_2, sv_1).data, )

    def test_waveplate(self, ):
        """
        test waveplate as a special case of a uniaxial crystal
        """
        thickness = np.random.rand() * 1e-2
        orientation = np.random.rand() * 360
        uni_crystal = UniaxialCrystal(
            orientation=orientation,
            thickness=thickness,
            cut_angle=0,
        )
        waveplate = Waveplate(
            orientation=orientation,
            thickness=thickness,
        )
        kwargs = {
            'wavelength': np.random.uniform(300e-9, 700e-9),
            'inc_angle': np.random.uniform(0, np.pi / 2),
            'azim_angle': np.random.uniform(0, 2 * np.pi),
        }
        print(uni_crystal.cut_angle)
        print(waveplate.cut_angle)
        assert_almost_equal(uni_crystal.get_delay(**kwargs), waveplate.get_delay(**kwargs))


if __name__ == '__main__':
    unittest.main()
