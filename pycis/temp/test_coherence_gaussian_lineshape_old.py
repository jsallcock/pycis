import numpy as np
import xarray as xr
import pycis
from scipy.constants import c

def test_with_gaussian_lineshape():
    """
    OLD CODE ALERT

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
    kappa_0 = pycis.get_kappa(wl_0, material, )
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
    doc_n1 = pycis.calculate_coherence(spectrum, delay_0, material=material)
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
    # np.testing.assert_almost_equal(doc_analytical(delay_0).real.ci_data_mast, doc_n2.real.ci_data_mast)

    titles = ['Spectrum', 'Contrast', 'Phase', ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
    plt.show()