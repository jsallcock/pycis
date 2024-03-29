import numpy as np
from numpy.fft import ifft2, ifftshift
import xarray as xr
from pycis.analysis import make_carrier_window, make_lowpass_window, fft2_im
from pycis.model import get_pixelated_phase_mask, get_pixel_idxs, get_superpixel_position


def demod_single_delay_pixelated(im):
    """
    :param im: xr.DataArray image to be demodulation. Must have dimensions 'x' and 'y'.
    :return:
    """
    sensor_format = [len(im.x), len(im.y)]
    idxs1, idxs2, idxs3, idxs4 = get_pixel_idxs(sensor_format)
    im = im.astype(float)
    xs, ys = get_superpixel_position(im.x, im.y, )
    m1 = im.isel(x=idxs1[0], y=idxs1[1], ).assign_coords({'x': xs, 'y': ys})
    m2 = im.isel(x=idxs2[0], y=idxs2[1], ).assign_coords({'x': xs, 'y': ys})
    m3 = im.isel(x=idxs3[0], y=idxs3[1], ).assign_coords({'x': xs, 'y': ys})
    m4 = im.isel(x=idxs4[0], y=idxs4[1], ).assign_coords({'x': xs, 'y': ys})
    m = np.arange(4)
    im = xr.concat([m1, m2, m3, m4], dim='m').assign_coords({'m': m, })
    i0 = im.sum(dim='m')
    phase = np.arctan2(im.sel(m=3) - im.sel(m=1), im.sel(m=0) - im.sel(m=2))
    contrast = 1 / i0 * np.sqrt(8 * np.power(im - i0 / 4, 2, ).sum(dim='m'))
    return i0 / 4, phase, contrast


def demod_single_delay_pixelated_mod(im):
    """
    alternative to demod_single_delay_pixelated() using 'synchronous demodulation' instead of the 'four-bucket' algorithm.
    
    :param im: xr.DataArray image to be demodulation. Must have dimensions 'x' and 'y'.
    :return:
    """
    if im.dims[0] != 'x':
        im = im.transpose('x', 'y')
    xs, ys = get_superpixel_position(im.x, im.y, )

    fft = fft2_im(im)
    pm = get_pixelated_phase_mask(im.shape)
    sp = im * np.exp(-1j * pm)
    fft_sp = fft2_im(sp)
    window_lowpass = make_lowpass_window(fft_sp, 100)
    fft_dc = fft * window_lowpass
    fft_carrier = fft_sp * window_lowpass
    dc = xr.DataArray(ifft2(ifftshift(fft_dc.data)), coords=im.coords, dims=im.dims).real
    carrier = xr.DataArray(ifft2(ifftshift(fft_carrier.data)), coords=im.coords, dims=im.dims)
    carrier *= 2
    phase = xr.ufuncs.angle(carrier)
    contrast = np.abs(carrier) / dc

    dc = dc.interp(x=xs, y=ys)
    phase = phase.interp(x=xs, y=ys)
    contrast = contrast.interp(x=xs, y=ys)
    return dc, phase, contrast


def demod_multi_delay_pixelated(image, fringe_freq, ):

    fft = fft2_im(image)
    window_pm = make_carrier_window(fft, fringe_freq, sign='pm')
    window_p = make_carrier_window(fft, fringe_freq, sign='p')
    window_m = make_carrier_window(fft, fringe_freq, sign='m')
    window_lowpass = make_lowpass_window(fft, fringe_freq)

    fft_carrier = fft * window_pm * window_lowpass
    fft_dc = (fft - fft_carrier) * window_lowpass

    fringe_freq_angle = np.arctan2(fringe_freq[1], fringe_freq[0])
    if np.pi / 4 <= fringe_freq_angle <= np.pi / 2:
        fft_carrier = fft_carrier.where(fft.freq_y < 0, 0) * 2
    else:
        fft_carrier = fft_carrier.where(fft.freq_x < 0, 0) * 2

    dc = xr.DataArray(ifft2(ifftshift(fft_dc.data)), coords=image.coords, dims=image.dims).real
    carrier_1 = xr.DataArray(ifft2(ifftshift(fft_carrier.data)), coords=image.coords, dims=image.dims)

    pm = get_pixelated_phase_mask(image.shape)
    sp = image * np.exp(1j * pm)

    fft_sp = fft2_im(sp)
    c, d = image.coords, image.dims
    carrier_3 = xr.DataArray(ifft2(ifftshift((fft_sp * window_p * window_lowpass).data)), coords=c, dims=d)
    carrier_4 = xr.DataArray(ifft2(ifftshift((fft_sp * window_m * window_lowpass).data)), coords=c, dims=d)
    carrier_2 = xr.DataArray(ifft2(ifftshift((fft_sp * window_lowpass).data)), coords=c, dims=d) - carrier_3 - carrier_4

    carrier_3 *= 8
    carrier_4 *= 8
    carrier_2 *= 4

    phase = []
    contrast = []
    for carrier in [carrier_1, carrier_2, carrier_3, carrier_4]:
        phase.append(-xr.ufuncs.angle(carrier))
        contrast.append(np.abs(carrier) / dc)


    import matplotlib.pyplot as plt

    plt.figure()
    window_lowpass.plot(x='freq_x', y='freq_y', )

    plt.figure()
    window_pm.plot(x='freq_x', y='freq_y',)

    plt.show()

    return dc, phase, contrast


def demod_triple_delay_pixelated(image, fringe_freq, **kwargs):
    """

    :param image: (xr.DataArray) CIS Image to demodulate
    :param fringe_freq: (tuple) Tuple containing xy-coord of fringe location in Fourier space
    :param kwargs: Additional kwargs - namely wfactor passed to make_carrier_window()
    :return:
    """
    fft = fft2_im(image)
    window_p = make_carrier_window(fft, fringe_freq, sign='p', **kwargs)
    window_m = make_carrier_window(fft, fringe_freq, sign='m', **kwargs)
    window_lowpass = make_lowpass_window(fft, fringe_freq)

    fft_dc = fft * window_lowpass

    dc = xr.DataArray(ifft2(ifftshift(fft_dc.data)), coords=image.coords, dims=image.dims).real

    pm = get_pixelated_phase_mask(image.shape)
    sp = image * np.exp(1j * pm)

    fft_sp = fft2_im(sp)
    c, d = image.coords, image.dims
    carrier_sum = xr.DataArray(ifft2(ifftshift((fft_sp * window_p * window_lowpass).data)), coords=c, dims=d)
    carrier_diff = xr.DataArray(ifft2(ifftshift((fft_sp * window_m * window_lowpass).data)), coords=c, dims=d)
    carrier_2 = xr.DataArray(ifft2(ifftshift((fft_sp * window_lowpass).data)), coords=c, dims=d) - carrier_sum - carrier_diff

    carrier_sum *= -8 / np.sqrt(2)
    carrier_diff *= 8 / np.sqrt(2)
    carrier_2 *= 4 / np.sqrt(2)

    phase = []
    contrast = []
    for carrier in [carrier_2, carrier_sum, carrier_diff]:
        phase.append(-xr.ufuncs.angle(carrier))
        contrast.append(np.abs(carrier) / dc)

    return dc, phase, contrast
