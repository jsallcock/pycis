import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import xarray as xr
from scipy.constants import c, atomic_mass, e
from pycis.model import calculate_coherence, get_kappa, get_spectrum_ciii_triplet


class TestCoherence(unittest.TestCase):
    def test_gaussian_multiplet(self, ):
        """
        Test that the correct coherence is returned for a simple spectral lineshape for which an analytical result
        exists.

        """
        # define spectrum (Doppler-broadened carbon III triplet at 464.9 nm):
        wls = np.array([464.742e-9, 465.025e-9, 465.147e-9, ])  # line component centre wavelengths in m
        rel_ints = np.array([0.556, 0.333, 0.111, ])  # relative intensities
        freq_0 = (c / wls * rel_ints).sum()  # centre-of-mass frequency in Hz
        temperature = 15  # in eV
        sigma_freq = freq_0 / c * np.sqrt(temperature * e / (12 * atomic_mass))  # line Doppler-width st. dev. in Hz

        delay = np.linspace(0, 40000, 100)  # randomly generate interferometer delays (in rad)
        delay = xr.DataArray(delay, dims=('delay_0',), coords=(delay, ), attrs={'units': 'rad'})

        materials = [None, 'a-BBO']
        kappas = [1, get_kappa(c / freq_0, material='a-BBO')]

        for material, kappa in zip(materials, kappas):

            def doc_gaussian(d, f_0, sigma):
                """
                Degree of coherence (DOC) of an area-normalised Gaussian function for delay d
                d is delay in rad, f_0 is centre-frequency and sigma is standard deviation, both in Hz
                """
                s = 1 + kappa * (f_0 - freq_0) / freq_0
                return np.exp(-2 * (np.pi * sigma * (d / (2 * np.pi * freq_0)) * kappa) ** 2) * np.exp(1j * d * s)

            doc_analytical = 0
            for wl, rel_int in zip(wls, rel_ints):
                doc_analytical += rel_int * doc_gaussian(delay, c / wl, sigma_freq)

            spectrum = get_spectrum_ciii_triplet(temperature, domain='frequency',)
            doc_numerical = calculate_coherence(spectrum, delay, material=material, freq_ref=freq_0, )

            assert_almost_equal(doc_numerical.real.data, doc_analytical.real.data, )
            assert_almost_equal(doc_numerical.imag.data, doc_analytical.imag.data, )


if __name__ == '__main__':
    unittest.main()
