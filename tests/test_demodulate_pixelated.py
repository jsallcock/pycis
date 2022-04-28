import os
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import xarray as xr
from pycis.model import get_pixelated_phase_mask, Instrument, get_spectrum_delta
from pycis.analysis import wrap, demod_triple_delay_pixelated
import matplotlib.pyplot as plt



class TestDemodulate(unittest.TestCase):

    def test_triple_delay_pixelated_demod(self, plot=True):
        """
        Test that the output of the 'demod_triple_delay_pixelated' close to expected result
        """

        inst = Instrument(config='triple_delay_pixelated.yaml')
        spectrum = get_spectrum_delta(465e-9, 5e3)
        igram = inst.capture(spectrum, clean=True)

        fringe_freq = inst.retarders[0].get_fringe_frequency(465e-9, inst.optics[-1])

        delays = inst.get_delay(465e-9, igram.x,igram.y)
        phase_mask = get_pixelated_phase_mask(igram.shape)

        delays_pm_wrapped = []

        for delay in delays:
            delays_pm_wrapped.append(wrap(delay))

        demod_data = demod_triple_delay_pixelated(igram, fringe_freq)

        assert_almost_equal(demod_data[1][0].values, delays_pm_wrapped[0].values, err_msg="Demod of phi_2 not working!")
        assert_almost_equal(demod_data[1][1].values, delays_pm_wrapped[1].values, err_msg="Demod of phi_sum not working!")
        assert_almost_equal(demod_data[1][2].values, delays_pm_wrapped[2].values, err_msg="Demod of phi_diff not working!")

        assert_almost_equal(demod_data[2][0].values, np.ones(igram.shape).values, err_msg="Demod of zeta_2 not working!")
        assert_almost_equal(demod_data[2][1].values, np.ones(igram.shape).values, err_msg="Demod of zeta_sum not working!")
        assert_almost_equal(demod_data[2][2].values, np.ones(igram.shape).values, err_msg="Demod of zeta_diff not working!")

if __name__ == '__main__':
    unittest.main()