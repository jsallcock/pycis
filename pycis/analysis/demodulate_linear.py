import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
import xarray as xr

from pycis.analysis import make_carrier_window


def fft2_im(image):
    """
    2-D fast Fourier transform of an image

    :param image:
    :return: fft
    """

    attrs = {'units': 'm^-1'}

    freq_x = fftshift(fftfreq(len(image.x), np.diff(image.x)[0]))
    freq_x = xr.DataArray(freq_x, dims=('freq_x',), coords={'freq_x': freq_x}, attrs=attrs)

    freq_y = fftshift(fftfreq(len(image.y), np.diff(image.y)[0]))
    freq_y = xr.DataArray(freq_y, dims=('freq_y',), coords={'freq_y': freq_y}, attrs=attrs)

    return xr.DataArray(fftshift(fft2(image.data)), coords=(freq_x, freq_y), )


def demodulate_linear(image, fringe_freq, ):
    """
    demodulation of interferograms with a linear phase shear

    :param image: xr.DataArray with dimensions 'x' and 'y' in units m, indicating pixel position on the sensor plane.
    :param fringe_freq: tuple / list of two floats corresponding to the x- and y-components of the predicted
    fringe frequency in units of m^(-1)
    :return:
    """

    fft = fft2_im(image)
    window = make_carrier_window(fft, fringe_freq)

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

    return dc, phase, contrast
