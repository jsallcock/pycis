import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr
import matplotlib.pyplot as plt


def make_carrier_window(fft, fringe_freq, sign='p'):
    """
    Generates Fourier-domain window to isolate a carrier term at the given spatial frequency.

    Window extends outwards in the orthogonal direction to the fringe frequency.

    :param fft: (xr.DataArray) Fourier-trasformed image with dimensions 'freq_x' and 'freq_y'
    :param fringe_freq:
    :param sign: (str) 'p' to window the positive frequency carrier term. 'm' to window the negative frequency carrier,
    'pm' to window both.
    :return: window (xr.DataArray) with same dims and coords as fft.
    """

    # make window for isolating carrier frequency
    fringe_freq_abs = np.sqrt(fringe_freq[0] ** 2 + fringe_freq[1] ** 2)
    fringe_freq_angle = np.arctan2(fringe_freq[1], fringe_freq[0])

    # define projected coordinate along direction of fringes
    coord = np.cos(fringe_freq_angle) * fft.freq_x + np.sin(fringe_freq_angle) * fft.freq_y
    condition_1 = coord < fringe_freq_abs - fringe_freq_abs / 1.5
    condition_2 = coord > fringe_freq_abs + fringe_freq_abs / 1.5
    window = xr.ufuncs.logical_not(condition_1 + condition_2).astype(float)

    window = xr.where(window == 0, np.nan, window)
    coord = window * coord
    coord = (coord - np.nanmin(coord)) / (np.nanmax(coord) - np.nanmin(coord))

    # manually define Blackman window
    alpha = 0.16
    a_0 = (1 - alpha) / 2
    a_1 = 0.5
    a_2 = alpha / 2
    window = a_0 - a_1 * np.cos(2 * np.pi * coord) + a_2 * np.cos(4 * np.pi * coord)
    window = xr.where(xr.ufuncs.isnan(window), 0, window, )
    window = xr.where(window < 0, 0, window, )

    if sign == 'p':
        pass
    elif sign == 'm':
        window.values = np.flip(window.values)
    elif sign == 'pm':
        window.values = window.values + np.flip(window.values)  # second window for negative frequency carrier
    else:
        raise Exception('input not understood')

    window /= float(window.max())  # normalise
    return window

def make_lowpass_window(fft, fringe_freq):
    """
    extremely quick and dirty for now

    # TODO think about this more

    :param fft:
    :param fringe_freq:
    :return:
    """

    window = xr.ones_like(fft.real)
    freq_x_lim, freq_y_lim = float(fft.freq_x.max() / 2), float(fft.freq_y.max() / 2)

    window = window.where((-freq_x_lim < window.freq_x) & (window.freq_x < freq_x_lim), 0)
    window = window.where((-freq_y_lim < window.freq_y) & (window.freq_y < freq_y_lim), 0)

    sigma = 0.00005 * np.array(fft.shape) * (abs(np.array(fringe_freq)) ** 0.7 / (4 * np.array([fft.freq_x.max(), fft.freq_y.max()]))) ** -1
    print(sigma)
    window.values = gaussian_filter(window.values, sigma)

    return window
