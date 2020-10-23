import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import xarray as xr
from scipy.constants import c
from pycis.model import calculate_coherence


class TestCoherence(unittest.TestCase):
    def test_gaussian_no_dispersion(self):
        """
        Test that the correct coherence is returned for a Gaussian lineshape
        """

        # define spectrum:
        wl_0 = 464.8e-9  # central wavelength in m
        wl_sigma = 0.05e-9  # width parameter in m
        n_sigma = 10  # extent of coordinate grid
        n_bins = 20000
        kappa_0 = 1  # this is a non-dispersive test

        # grid in frequency-space (in Hz)
        freq_0 = c / wl_0
        freq_sigma = c / wl_0 ** 2 * wl_sigma
        freq = np.linspace(freq_0 - n_sigma * freq_sigma, freq_0 + n_sigma * freq_sigma, n_bins)
        freq = xr.DataArray(freq, dims=('frequency',), coords=(freq,), attrs={'units': 'Hz'})

        spectrum = 1 / (freq_sigma * np.sqrt(2 * np.pi)) * np.exp(- 1 / 2 * ((freq - freq_0) / freq_sigma) ** 2)

        # randomly generate interferometer delay values (in radians)
        delay = np.random.rand(10) * 1e4
        delay = xr.DataArray(delay, dims=('delay_0',), coords=(delay, ), attrs={'units': 'rad'})

        doc_numerical = calculate_coherence(spectrum, delay, material=None)
        doc_analytical = np.exp(-2 * (np.pi * freq_sigma * (delay / (2 * np.pi * freq_0)) * kappa_0) ** 2) * \
                         np.exp(1j * delay)

        assert_almost_equal(doc_numerical.data, doc_analytical.data)


if __name__ == '__main__':
    unittest.main()
