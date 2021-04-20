import numpy as np
from numpy.fft import ifft2, ifftshift
import xarray as xr

from pycis.analysis import make_carrier_window, make_lowpass_window, fft2_im
from pycis.model import get_pixelated_phase_mask


def demodulate_single_delay_pixelated(img):
    """

    :param img:
    :return:
    """

    img = img.astype(float)  # unsigned integers can cause headaches

    # assert 'x' in img.dims and 'y' in img.dims
    # assert img.dims.index('x') + 1 == img.dims.index('y')
    img = img.drop(list(img.coords.variables.mapping.keys()))

    m1 = img[::2, ::2]
    m2 = img[1::2, ::2]
    m3 = img[1::2, 1::2]
    m4 = img[::2, 1::2]

    m = np.arange(4) + 1
    img = xr.concat([m1, m2, m3, m4], dim='m').assign_coords({'m': m, })
    i0 = img.sum(dim='m')
    phase = np.arctan2(img.sel(m=4) - img.sel(m=2), img.sel(m=3) - img.sel(m=1))
    contrast = 1 / i0 * np.sqrt(8 * np.power(img - i0 / 4, 2, ).sum(dim='m'))

    return i0 / 4, phase, contrast


def demodulate_multi_delay_pixelated(image, fringe_freq, ):

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
    carrier_2 *= 2

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