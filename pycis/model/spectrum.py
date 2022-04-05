import numpy as np
import xarray as xr
from scipy.constants import c,e, atomic_mass
from pycis.temp.zeeman import zeeman

"""
simple example spectra for testing
"""


def get_spectrum_delta(wl0, ph, ):
    """
    :param float wavelength: in metres.
    :param float ph: total photon fluence
    :return:
    """
    dwl = 1e-13
    wavelength = np.linspace(wl0 - dwl, wl0 + dwl, 3)
    wavelength = xr.DataArray(wavelength, coords=(wavelength, ), dims=('wavelength', ), )
    spectrum = xr.DataArray([0., 1., 0.], coords=(wavelength, ), dims=('wavelength', ))
    return spectrum * ph / spectrum.integrate(coord='wavelength')


def get_doppler_broadened_singlet (temperature, wl, mass, domain='frequency', nbins=1000,):
    """
    return area-normalised spectrum of the Doppler-broadened C III triplet at 464.9 nm.

    Used for testing.

    :param temperature: (float) in eV
    :param wavelength: (float) wavelength of the singlet peak in m
    :param mass: (float) ion mass in atomic mass units.
    :param domain: (str) 'frequency' or 'wavelength'
    :return: (xr.DataArray) spectrum
    """

    freq = c / wl
    sigma_freq = freq / c * np.sqrt(temperature * e / (mass * atomic_mass))  # line Doppler-width st. dev. in Hz

    n_sigma = 30  # extent of coordinate grid

    freq_axis = np.linspace(freq - n_sigma * sigma_freq, freq + n_sigma * sigma_freq, nbins)
    wavelength = c / freq_axis

    freq_axis = xr.DataArray(freq_axis, dims=('frequency',), coords=(freq_axis,), attrs={'units': 'Hz'})

    # area-normalised Gaussian function for frequency f
    def gaussian(f, f_0, sigma):
        """
        f is frequency, f_0 is centre-frequency and sigma is standard deviation, all in Hz
        """
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- 1 / 2 * ((f - f_0) / sigma) ** 2)

    spectrum = gaussian(freq_axis, c / wl, sigma_freq)

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


def get_spectrum_ciii_triplet(temperature, bfield=0, view=0, domain='frequency', nbins=1000, stokes=False, test=False):
    """
    return area-normalised spectrum of the Doppler-broadened C III triplet at 464.9 nm.

    Used for testing.

    :param temperature: (float) in eV
    :param domain: (str) 'frequency' or 'wavelength'
    :keyword stokes: if True, returns a stokes vector rather than a single intensity value.
    :return: (xr.DataArray) spectrum
    """

    stokes=stokes
    if test == True and bfield != 0:
        wls = np.array([4.648719972899276e-07 + 2.0213e-11 * bfield, 4.648719972899276e-07 - 2.0213e-11 * bfield,
                        4.6515488591713053e-07 + 2.0213e-11 * bfield, 4.6515488591713053e-07 - 2.0213e-11 * bfield,
                        4.652776012198098e-07 + 2.0213e-11 * bfield,  4.652776012198098e-07 - 2.0213e-11 * bfield])
        rel_ints = np.array([0.556/2, 0.556/2, 0.333/2, 0.333/2, 0.111/2, 0.111/2])  # relative intensities

    # define spectrum (corresponds to Doppler-broadened carbon III triplet at 464.9 nm)
    # if stokes is True, ensure that a stokes vector is returned, even if the magnetic field is 0.
    elif stokes is True:
        wls, s_vectors = zeeman(bfield=bfield, view=view, stokes=True)
        wls=np.array(wls)
        s_vectors = np.array(s_vectors)
        rel_ints = s_vectors[:,0]

    # otherwise, check that a magnetic field is present before accounting for zeeman splitting:
    elif bfield != 0 or bfield ==0:
        wls, rel_ints = zeeman(bfield=bfield, view=view, )
        wls = np.array(wls)
        rel_ints = np.array(rel_ints)

    # if just the intensities are required and there is no field, simply use the accepted values for wl and int.
    else:
         wls = np.array([464.742e-9, 465.025e-9, 465.147e-9, ])  # line component centre wavelengths in m
         rel_ints = np.array([0.556, 0.333, 0.111, ])  # relative intensities
    freqs = c / wls
    freq_com = (c / wls * rel_ints).sum()  # centre-of-mass frequency in Hz
    sigma_freq = freq_com / c * np.sqrt(temperature * e / (12 * atomic_mass))  # line Doppler-width st. dev. in Hz

    n_sigma = 30  # extent of coordinate grid

    freq = np.linspace(freqs.min() - n_sigma * sigma_freq, freqs.max() + n_sigma * sigma_freq, nbins)
    wavelength = c / freq

    freq = xr.DataArray(freq, dims=('frequency',), coords=(freq,), attrs={'units': 'Hz'})

    # area-normalised Gaussian function for frequency f
    def gaussian(f, f_0, sigma):
        """
        f is frequency, f_0 is centre-frequency and sigma is standard deviation, all in Hz
        """
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- 1 / 2 * ((f - f_0) / sigma) ** 2)

    spectrum = 0
    if stokes is False:
        for wl, rel_int in zip(wls, rel_ints):
            spectrum += rel_int * gaussian(freq, c / wl, sigma_freq)

    else:
        for wl, s_vector in zip(wls, s_vectors):
            print(wl, s_vector)
            s_vector = xr.DataArray(s_vector, dims='stokes')
            spectrum += s_vector * gaussian(freq, c / wl, sigma_freq)
            print(spectrum)

    if domain == 'frequency':
        return spectrum
    elif domain == 'wavelength':
        # convert from frequency to wavelength
        wlstr = 'wavelength'
        spectrum = spectrum.rename({'frequency': wlstr}).assign_coords({wlstr: wavelength}).sortby(wlstr, )
        if stokes is True:
            spectrum /= spectrum[0].integrate(coord='wavelength')
        else:
            spectrum /= spectrum.integrate(coord='wavelength')
        return spectrum
    else:
        raise Exception('input not understood')


def get_spectrum_cii_multiplet(temperature, domain='frequency', nbins=1000):
    """
    return area-normalised spectrum of the Doppler-broadened C II multiplet at 514.2 nm.

    Used for testing.

    :param temperature: (float) in eV
    :param domain: (str) 'frequency' or 'wavelength'
    :return: (xr.DataArray) spectrum
    """

    # define spectrum (corresponds to Doppler-broadened carbon III triplet at 464.9 nm):
    wls = np.array([513.295e-9, 513.328e-9, 513.726e-9, 513.917e-9,
                    514.350e-9, 514.517e-9, 515.109e-9])  # line component centre wavelengths in m
    rel_ints = np.array([0.140, 0.151, 0.029, 0.045, 0.139, 0.349, 0.149])  # relative intensities
    freqs = c / wls
    freq_com = (c / wls * rel_ints).sum()  # centre-of-mass frequency in Hz
    sigma_freq = freq_com / c * np.sqrt(temperature * e / (12 * atomic_mass))  # line Doppler-width st. dev. in Hz

    n_sigma = 30  # extent of coordinate grid

    freq = np.linspace(freqs.min() - n_sigma * sigma_freq, freqs.max() + n_sigma * sigma_freq, nbins)
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
