import numpy as np
import xarray as xr


def demodulate_ppm(image):
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