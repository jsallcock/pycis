import numpy as np
from numba import vectorize, float64, complex128
import xarray as xr
import matplotlib.pyplot as plt
from scipy.constants import c
import pycis


def calculate_coherence(spectrum, delay, material=None, freq_com=None):
    """
    Calculates the (temporal) coherence of a given intensity spectrum for a given interferometer delay(s).


    Temporal coherence :math:`\\Gamma(\\tau)` is the Fourier transform of the frequency spectrum :math:`I(\\nu)`:

     .. math::
        \\Gamma(\\tau)=\\int_{-\\infty}^{\\infty}I(\\nu)\\exp(2\\pi{}i\\nu\\tau)d\\nu,

    with interferometer delay time :math:`\\tau` and frequency :math:`\\nu` as the conjugate variables.
    Generally, dispersion means that :math:`\\tau\\rightarrow\\tau(\\nu)`  but a first-order (linear) approx. for dispersion maintains it in a modified form (the 'group delay' approximation).

    :param xr.DataArray spectrum: Intensity spectrum. Dim. 'wavelength' has coords with units m or else dim.
        'frequency' has coords with units Hz. Spectrum units are then either ( arb. / m ) or (arb. / Hz ) respectively.
    :param xr.DataArray delay: Interferometer delay in units radians. If delay is a float or is a DataArray without a
        'wavelength' or a 'frequency' dimension, then the group delay approx. is used. In this case, it is assumed that
        the delay value(s) correspond to the centre-of-mass (COM) frequency of the given spectrum and the coherence is
        calculated for each delay value. If delay has either a 'wavelength' dim. or a 'frequency' dim. -- with
        coordinates that match the corresponding dim. of spectrum -- then the full dispersive integral is evaluated.
    :param material: string specifying the interferometer crystal material. See pycis.model.dispersion for valid inputs.
         This is not needed in the full dispersive treatment as the dispersion info has already been provided in the
         delay argument. To do a none-dispersive calculation, you should leave material=None and use a delay argument
         that will trigger the group delay approx.
    :param freq_com: centre of mass frequency of spectrum, if it has already been calculated.
    :return: coherence (temporal). Units are those of the spectrum argument, but integrated over the spectral dimension e.g. if spectrum has units ( W / m^2 / m ) then coherence has units ( W / m^2 ).
    """

    # if necessary, convert spectrum's wavelength (m) dim + coordinate to frequency (Hz)
    if 'wavelength' in spectrum.dims:
        spectrum = spectrum.rename({'wavelength': 'frequency'})
        spectrum['frequency'] = c / spectrum['frequency']
        spectrum = spectrum.sortby('frequency')
        spectrum /= spectrum.integrate(coord='frequency')

    # calculate centre of mass (c.o.m.) frequency if not supplied
    if freq_com is None:
        freq_com = (spectrum * spectrum['frequency']).integrate(coord='frequency') / \
                   spectrum.integrate(coord='frequency')

    # determine calculation mode
    if hasattr(delay, 'dims'):
        if 'frequency' in delay.dims or 'wavelength' in delay.dims:
            mode = 'full_dispersive'
        else:
            mode = 'group_delay'
    else:
        mode = 'group_delay'

    if mode == 'full_dispersive':
        # if necessary, convert delay's wavelength dim + coordinate to frequency
        if 'wavelength' in delay.dims:
            delay = delay.rename({'wavelength': 'frequency'})
            delay['frequency'] = c / delay['frequency']
        integrand = spectrum * complexp_ufunc(delay)

    elif mode == 'group_delay':
        if material is not None:
            kappa_0 = pycis.calculate_kappa(c / freq_com, material=material, )
        else:
            kappa_0 = 1

        freq_shift_norm = (spectrum['frequency'] - freq_com) / freq_com
        integrand = spectrum * complexp_ufunc(delay * (1 + kappa_0 * freq_shift_norm))
    else:
        raise NotImplementedError

    integrand = integrand.sortby(integrand.frequency)  # ensure that integration limits are from -ve to +ve frequency
    return integrand.integrate(coord='frequency')


@vectorize([complex128(float64)], fastmath=False, nopython=True, cache=True, )
def complexp(x):
    return np.exp(1j * x)


def complexp_ufunc(x):
    return xr.apply_ufunc(complexp, x, dask='allowed', )


def test_with_gaussian_lineshape():
    """
    numerical / analytical test of calculate_coherence() using a modelled Gaussian spectral lineshape

    TODO move elsewhere

    old. abbreviations:
    doc = degree of coherence
    wl = wavelength
    biref = birefringence

    :return:
    """
    import time

    # ANALYTICAL
    wl_0 = 464.8e-9
    material = 'a-BBO'
    kappa_0 = pycis.calculate_kappa(wl_0, material, )
    wl_sigma = 0.05e-9
    n_sigma = 10
    n_bins = 20000
    thickness = np.array([
        4.48e-3,
        6.35e-3,
        9.79e-3,
    ])

    # calculate delays at wl_0 for the given waveplate thicknesses
    biref_0 = pycis.calculate_dispersion(wl_0, material, )[0]
    delay_0 = abs(2 * np.pi * thickness * biref_0 / wl_0)  # (rad)
    delay_0 = xr.DataArray(delay_0, dims=('delay_0',), coords=(delay_0,), attrs={'units': 'rad'})

    # generate spectrum in frequency-space
    freq_0 = c / wl_0
    freq_sigma = c / wl_0 ** 2 * wl_sigma
    freq = np.linspace(freq_0 - n_sigma * freq_sigma, freq_0 + n_sigma * freq_sigma, n_bins)[::-1]
    freq = xr.DataArray(freq, dims=('frequency', ), coords=(freq, ), attrs={'units': 'Hz'})
    spectrum = 1 / (freq_sigma * np.sqrt(2 * np.pi)) * np.exp(- 1 / 2 * ((freq - freq_0) / freq_sigma) ** 2)

    delay_time_axis_0 = np.linspace(0, n_sigma / 20 * 1 / freq_sigma, n_bins)
    delay_axis_0 = 2 * np.pi * delay_time_axis_0 * freq_0
    delay_axis_0 = xr.DataArray(delay_axis_0, coords=(delay_axis_0, ), dims=('delay_0', ), attrs={'units': 'rad'}, )

    def doc_analytical(delay_0):
        doc_analytical = np.exp(-2 * (np.pi * freq_sigma * (delay_0 / (2 * np.pi * freq_0)) * kappa_0) ** 2) * \
                         np.exp(1j * delay_0)
        return doc_analytical

    # NUMERICAL 1 (n1) -- tests group delay approximation
    s = time.time()
    doc_n1 = calculate_coherence(spectrum, delay_0, material=material)
    e = time.time()
    print('numerical_1 (group delay approx.):', e - s, 'seconds')

    # NUMERICAL 2 (n2) -- tests full dispersion integral
    thickness = xr.DataArray(thickness, dims=('delay_0', ), coords=(delay_0, ))
    biref = pycis.calculate_dispersion(c / freq, material)[0]
    delay = abs(2 * np.pi * biref * thickness / (c / freq))

    s = time.time()
    doc_n2 = calculate_coherence(spectrum, delay, material=material)
    e = time.time()
    print('numerical_2 (full dispersion calc.):', e - s, 'seconds')

    # PLOT
    fig = plt.figure(figsize=(10, 5,))
    axes = [fig.add_subplot(1, 3, i) for i in [1, 2, 3, ]]

    # plot spectrum
    spectrum.plot(ax=axes[0])

    # plot contrast and phase
    funcs = [np.abs, xr.ufuncs.angle]
    for idx, (func, ax, ) in enumerate(zip(funcs, axes[1:])):
        if idx == 0:
            func(doc_analytical(delay_axis_0)).plot(ax=ax, lw=0.5, color='C0')
        func(doc_analytical(delay_0)).plot(ax=ax, lw=0, marker='x', markersize=12,
                                             label='Analytical\n(Group delay approx.)', color='C0')
        func(doc_n1).plot(ax=ax, lw=0, marker = '.', markeredgewidth=0.5, markeredgecolor='k',
                              label='Numerical\n(Group delay approx.)', markersize=8, color='C1')
        func(doc_n2).plot(ax=ax, lw=0, marker='d', markersize=12, fillstyle='none', label='Numerical (full)', color='C2')
        leg = ax.legend(fontsize=7)


    np.testing.assert_almost_equal(doc_analytical(delay_0).real.data, doc_n1.real.data)
    # np.testing.assert_almost_equal(doc_analytical(delay_0).real.data, doc_n2.real.data)

    titles = ['Spectrum', 'Contrast', 'Phase', ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    test_with_gaussian_lineshape()
