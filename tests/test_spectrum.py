import unittest

from numpy.testing import assert_almost_equal
from scipy.constants import c
from pycis.model import get_spectrum_doppler_singlet, wl2freq, freq2wl
import matplotlib.pyplot as plt


class TestSpectrum(unittest.TestCase):
    def test_domain_switch(self, ):
        """
        Test that we are converting spectra from wavelength domain to frequency domain correctly.
        """
        kwargs = {
            'temperature': 5,
            'wl0': 465e-9,
            'mass': 12,
            'v': 0,
            'nbins': 5000,
            'nsigma': 10,
        }
        spec_freq = get_spectrum_doppler_singlet(**kwargs, domain='frequency')
        spec_wl = freq2wl(spec_freq)

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # spec_freq.plot(x='frequency', ax=ax)
        #
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # spec_wl.plot(x='wavelength', ax=ax)
        #
        # plt.show()

        assert_almost_equal(spec_freq.integrate(coord='frequency'), spec_wl.integrate(coord='wavelength'))
        assert_almost_equal(c / spec_freq.frequency.values[::-1], spec_wl.wavelength.values)


if __name__ == '__main__':
    unittest.main()
