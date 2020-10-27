import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
import xarray as xr

from pycis.analysis import make_carrier_window, make_lowpass_window, fft2_im


def demodulate_single_delay_polarised(image):
    """

    :param image:
    :return:
    """

    x_superpixel = np.arange(image.shape[0] / 2)
    y_superpixel = np.arange(image.shape[1] / 2)

    assert 'x' in image.dims and 'y' in image.dims
    image = image.drop(list(image.coords.variables.mapping.keys()))

    m1 = image[::2, ::2]
    m2 = image[1::2, ::2]
    m3 = image[1::2, 1::2]
    m4 = image[::2, 1::2]

    m = np.arange(4) + 1
    image = xr.concat([m1, m2, m3, m4], dim='m').assign_coords({'m': m, 'x': x_superpixel, 'y': y_superpixel, })
    i0 = image.sum(dim='m')
    phase = np.arctan2(image.sel(m=4) - image.sel(m=2), image.sel(m=3) - image.sel(m=1))
    contrast = 1 / i0 * np.sqrt(8 * np.power(image - i0 / 4, 2, ).sum(dim='m'))

    return i0, phase, contrast


def demodulate_multi_delay_polarised(image, fringe_freq, ):

    fft = fft2_im(image)
    window = make_carrier_window(fft, fringe_freq)
    w = make_lowpass_window(fft, fringe_freq)

    fft_dc = fft * (1 - window)
    fft_carrier = fft * window

    fringe_freq_angle = np.arctan2(fringe_freq[1], fringe_freq[0])
    if np.pi / 4 <= fringe_freq_angle <= np.pi / 2:
        fft_carrier = fft_carrier.where(fft.freq_y < 0, 0) * 2
    else:
        fft_carrier = fft_carrier.where(fft.freq_x < 0, 0) * 2

    dc = xr.DataArray(ifft2(ifftshift(fft_dc.data)), coords=image.coords, dims=image.dims).real
    carrier = xr.DataArray(ifft2(ifftshift(fft_carrier.data)), coords=image.coords, dims=image.dims)

    phase = -xr.ufuncs.angle(carrier)  # negative sign to match with modelling conventions
    contrast = np.abs(carrier) / dc
    return 5, phase, contrast