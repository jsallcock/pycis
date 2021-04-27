import numpy as np
import xarray as xr
"""
functions for generating example spectra -- mostly for testing.
"""

def get_spectrum_delta(wl0, ph, ):
    """
    :param float wavelength: in metres.
    :param float ph:
    :return:
    """
    dwl = 1e-13
    wavelength = np.linspace(wl0 - dwl, wl0 + dwl, 3)
    wavelength = xr.DataArray(wavelength, coords=(wavelength, ), dims=('wavelength', ), )
    spectrum = xr.DataArray([0., 1., 0.], coords=(wavelength, ), dims=('wavelength', ))
    return spectrum * ph / spectrum.integrate(coord='wavelength')


def get_spectrum_ciii_triplet(temperature, domain='frequency', nbins=1000):
    """
    return area-normalised spectrum of the Doppler-broadened C III triplet at 464.9 nm.

    Used for testing.

    :param temperature: (float) in eV
    :param domain: (str) 'frequency' or 'wavelength'
    :return: (xr.DataArray) spectrum
    """

    # define spectrum (corresponds to Doppler-broadened carbon III triplet at 464.9 nm):
    wls = np.array([464.742e-9, 465.025e-9, 465.147e-9, ])  # line component centre wavelengths in m
    rel_ints = np.array([0.556, 0.333, 0.111, ])  # relative intensities
    freq_com = (c / wls * rel_ints).sum()  # centre-of-mass frequency in Hz
    sigma_freq = freq_com / c * np.sqrt(temperature * e / (12 * atomic_mass))  # line Doppler-width st. dev. in Hz

    n_sigma = 30  # extent of coordinate grid

    freq = np.linspace(freq_com - n_sigma * sigma_freq, freq_com + n_sigma * sigma_freq, nbins)
    wavelength = c / freq

    freq = xr.DataArray(freq, dims=('frequency',), coords=(freq,), attrs={'units': 'Hz'})

    # area-normalised Gaussian function for frequency f
    def gaussian(f, f_0, sigma):
        """
        f is frequency, f_0 is centre-frequency and sigma is standard deviation, all in Hz
        """
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- 1 / 2 * ((f - f_0) / sigma) ** 2)

    spectrum = 0
    for wl, rel_int in zip(wls, rel_ints):
        spectrum += rel_int * gaussian(freq, c / wl, sigma_freq)

    if domain == 'frequency':
        return spectrum
    elif domain == 'wavelength':
        # convert from frequency to wavelength
        wlstr = 'wavelength'
        spectrum = spectrum.rename({'frequency': wlstr}).assign_coords({wlstr: wavelength}).sortby(wlstr, )
        spectrum /= spectrum.integrate(coord='wavelength')
        return spectrum
    else:
        raise Exception('input not understood')


if __name__ == '__main__':
    sp = get_spectrum_delta(465e-9, 1.e4)