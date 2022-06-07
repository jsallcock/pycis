from typing import Union
import numpy as np
from numba import vectorize, float64, complex128
import xarray as xr
from scipy.constants import c
from pycis.model import get_kappa, wl2freq


def calculate_coherence(spectrum, delay, material=None, freq_ref=None):
    """
    Calculate the temporal coherence of an intensity spectrum, as measured by a 2-beam interferometer with given delay(s).


    Temporal coherence :math:`\\Gamma(\\tau)` is the Fourier transform of the frequency spectrum :math:`I(\\nu)`:

     .. math::
        \\Gamma(\\tau)=\\int_{-\\infty}^{\\infty}I(\\nu)\\exp(2\\pi{}i\\nu\\tau)d\\nu,

    with interferometer delay time :math:`\\tau` and frequency :math:`\\nu` as the conjugate variables. It is measured
    using a 2-beam interferometer. In practice, instrument dispersion is present: :math:`\\tau\\rightarrow\\tau(\\nu)`.
    How this dispersive integral is evaluated by this function depends on the arguments given. For a full explanation of
    how dispersion affects the coherence measured by an interferometer, see Section 2.2.2 of J. Allcock's PhD thesis.

    :param spectrum: \
        Intensity spectrum as a DataArray. Dimension 'wavelength' has coordinates with units m or else dimension
        'frequency' has coordinates with units Hz. Units of spectrum are then ( arb. / m ) or (arb. / Hz )
        respectively. This function broadcasts across xr.DataArray dimensions, so spectrum can represent e.g. a
        'spectral cube' with dimensions 'x' and 'y'.
    :type spectrum: xr.DataArray

    :param delay: Interferometer delay in radians. Type determines how the calculation is performed :

           - Mode 1: No dispersion. If delay is a scalar, or a DataArray without a spectral dimension ('wavelength' /
             'frequency'), and material = None then the calculation assumes no dispersion. Delay(s) correspond to
             the reference frequency.
           - Mode 2: Group delay approximation. If delay is a scalar, or a DataArray without a spectral dimension,
             and material != None then the 'group delay approximation' for dispersion is used. This is a first-order
             Taylor expansion of delay about the reference frequency. Delay(s) correspond to the reference frequency.
           - Mode 3: Full dispersive integral. If delay is a DataArray with a spectral dimension whose coordinates match
             those of spectrum, then the full dispersive integral can be evaluated. Mostly here for testing.

    :type delay: Union[float, xr.DataArray]

    :param material: \
        String specifying the interferometer crystal material, which determines the dispersion.
        See pycis.model.dispersion for valid strings. Only used in 'Mode 2' (see above). Defaults to material = None,
        giving either a non-dispersive calculation or a full dispersive calculation, depending on the delay argument.
    :type material: str

    :param freq_ref: \
        Reference frequency to which the delay argument corresponds (if it is scalar or a DataArray without a
        spectral dimension). Only used for modes 1 & 2. The rest-frame centre-of-mass
        frequency of the spectral feature being studied is typically a sensible choice. Defaults to the centre-of-mass
        frequency of the given spectrum.
    :type freq_ref: Union[float, xr.DataArray]

    :return: Temporal coherence. Units are those of the spectrum argument, but integrated over the spectral dimension
        e.g. if spectrum has units ( W / m^2 / m ) then coherence has units ( W / m^2 ). If spectrum is area normalised
        to one then the temporal coherence is the unitless 'degree of temporal coherence'.
    """

    # calculate in frequency domain
    if 'wavelength' in spectrum.dims:
        assert 'frequency' not in spectrum.dims
        spectrum = wl2freq(spectrum)

    # determine calculation mode
    mode = None
    if isinstance(delay, xr.DataArray):
        if 'frequency' in delay.dims or 'wavelength' in delay.dims:
            mode = 'full_dispersive'
    if mode is None:
        if material is None:
            mode = 'no_dispersion'
        else:
            mode = 'group_delay'

    if mode == 'full_dispersive':
        # if necessary, convert delay's wavelength dim + coordinate to frequency
        if 'wavelength' in delay.dims:
            delay = delay.rename({'wavelength': 'frequency'})
            delay['frequency'] = c / delay['frequency']
        integrand = spectrum * complexp_ufunc(delay)

    else:
        if freq_ref is None:
            freq_ref = (spectrum * spectrum['frequency']).integrate(coord='frequency') / \
                       spectrum.integrate(coord='frequency')

        if mode == 'group_delay':
            kappa = get_kappa(c / freq_ref, material=material)
        elif mode == 'no_dispersion':
            kappa = 1

        freq_shift_norm = (spectrum['frequency'] - freq_ref) / freq_ref
        integrand = spectrum * complexp_ufunc(delay * (1 + kappa * freq_shift_norm))

    integrand = integrand.sortby(integrand.frequency)  # ensures integration limits are from -ve to +ve frequency
    return integrand.integrate(coord='frequency')


@vectorize([complex128(float64)], fastmath=False, nopython=True, cache=True, )
def complexp(x):
    return np.exp(1j * x)


def complexp_ufunc(x):
    return xr.apply_ufunc(complexp, x, dask='allowed', )


def get_coherence_doppler_singlet(temperature, wl0, mass, v, delay=None, material=None, nbins=500, nsigma=5):
    """

    :param temperature:
    :param wl0: assumed
    :param mass:
    :param v:
    :return:
    """

    # freq0 = c / wl0
    # freq_shift_norm = (f_0 - freq_0) / freq_0
    #
    # if mode == 'group_delay':
    #     kappa = get_kappa(wl0, material=material)
    # elif mode == 'no_dispersion':
    #     kappa = 1
    #
    # phase = delay * (1 + kappa * freq_shift_norm)
    # contrast = np.exp(-2 * (np.pi * sigma * (d / (2 * np.pi * freq_0)) * kappa) ** 2)

    return

