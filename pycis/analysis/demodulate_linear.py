import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
import xarray as xr
import matplotlib.pyplot as plt


def demodulate_linear(image, fringe_freq, ):
    """
    demodulation of interferograms with a linear phase shear

    :param image: xr.DataArray with dimensions 'x' and 'y' in units m, indicating pixel position on the sensor plane.
    :param fringe_freq: tuple / list of two floats corresponding to the x- and y-components of the predicted
    fringe frequency in units of m^(-1)
    :return:
    """

    freq_x = fftshift(fftfreq(len(image.x), np.diff(image.x)[0]))
    freq_x = xr.DataArray(freq_x, dims=('freq_x',), coords={'freq_x': freq_x})
    freq_y = fftshift(fftfreq(len(image.y), np.diff(image.y)[0]))
    freq_y = xr.DataArray(freq_y, dims=('freq_y',), coords={'freq_y': freq_y})

    fft = xr.DataArray(fftshift(fft2(image.data)), coords=(freq_x, freq_y), )

    # make window for isolating carrier frequency
    fringe_freq_abs = np.sqrt(fringe_freq[0] ** 2 + fringe_freq[1] ** 2)
    fringe_freq_angle = np.arctan2(fringe_freq[1], fringe_freq[0])

    coord_proj = np.cos(fringe_freq_angle) * freq_x + np.sin(fringe_freq_angle) * freq_y
    condition_1 = coord_proj < fringe_freq_abs - fringe_freq_abs / 1.5
    condition_2 = coord_proj > fringe_freq_abs + fringe_freq_abs / 1.5
    window = xr.ufuncs.logical_not(condition_1 + condition_2).astype(float)

    window = xr.where(window == 0, np.nan, window)
    coord_proj = window * coord_proj
    coord_proj = (coord_proj - np.nanmin(coord_proj)) / (np.nanmax(coord_proj) - np.nanmin(coord_proj))

    # Blackman window
    alpha = 0.16
    a_0 = (1 - alpha) / 2
    a_1 = 0.5
    a_2 = alpha / 2
    window = a_0 - a_1 * np.cos(2 * np.pi * coord_proj) + a_2 * np.cos(4 * np.pi * coord_proj)
    window = xr.where(xr.ufuncs.isnan(window), 0, window, )
    window = xr.where(window < 0, 0, window, )

    window.values = window.values + np.flip(window.values)  # second window for negative frequency carrier
    window /= float(window.max())  # normalise

    fft_dc = fft * (1 - window)
    fft_carrier = fft * window

    if np.pi / 4 <= fringe_freq_angle <= np.pi / 2:
        fft_carrier = fft_carrier.where(freq_y < 0, 0) * 2
    else:
        fft_carrier = fft_carrier.where(freq_x < 0, 0) * 2

    dc = xr.DataArray(ifft2(ifftshift(fft_dc.data)), coords=image.coords, dims=image.dims).real
    carrier = xr.DataArray(ifft2(ifftshift(fft_carrier.data)), coords=image.coords, dims=image.dims)

    phase = -xr.ufuncs.angle(carrier)  # negative sign to match with modelling conventions
    contrast = np.abs(carrier) / dc

    return dc, phase, contrast
