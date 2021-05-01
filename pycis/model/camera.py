import inspect
import numpy as np
import xarray as xr
from pycis.model import LinearPolariser, mueller_product

camera_types = [
    'monochrome',
    'monochrome_polarised',
    'rgb',
]


class Camera(object):
    """
    :param tuple sensor_format: Number of pixels in each dimension (x, y, ).
    :param float pix_size: Pixel size in m.
    :param int bit_depth: Bit depth of sensor. Currently limited to 16.
    :param float qe: Quantum efficiency or sensor in units e- per photon.
    :param float epercount: Conversion gain of sensor in units e- per count.
    :param float cam_noise: Camera noise standard deviation in units e-.
    :param str type: Describes the mode of sensor operation. Currently supported are: 'monochrome', 'rgb'
        (untested) for color imaging and 'monochrome_polarised' for monochrome with a pixelated polariser array
        (currently assume the 2x2 polariser layout of the FLIR Blackfly S).
    """
    def __init__(self, sensor_format, pixel_size, bit_depth, qe, epercount, cam_noise, type='monochrome'):
        self.sensor_format = sensor_format
        self.pixel_size = pixel_size
        self.bit_depth = bit_depth
        self.qe = qe
        self.epercount = epercount
        self.cam_noise = cam_noise
        self.type = type
        self.x, self.y = self.get_pixel_position()

        assert type in camera_types
        if type == 'monochrome_polarised':
            assert sensor_format[0] % 2 == 0
            assert sensor_format[1] % 2 == 0

    def capture(self, spectrum, apply_polarisers=None, clean=False):
        """
        Capture an image.

        :param xr.DataArray spectrum: Spectrum in units of photons / m with dimensions 'x', 'y', 'wavelength' and
            (optionally) 'stokes'. If there is no 'stokes' dim then the light is assumed to be unpolarised.
        :param bool apply_polarisers: Whether to apply a pixelated polariser array to the spectrum.
        :param bool clean: False to add realistic image noise. Clean images used for testing.
        :return: (xr.DataArray) Captured image.
        """

        if apply_polarisers is None:
            if self.type == 'monochrome_polarised':
                apply_polarisers = True

        if apply_polarisers:
            assert 'stokes' in spectrum.dims
            mueller_matrix = self.get_mueller_matrix()
            spectrum = mueller_product(mueller_matrix, spectrum, )

        # ensure only total intensity (first Stokes parameter) is observed
        if 'stokes' in spectrum.dims:
            spectrum = spectrum.isel(stokes=0, drop=True)

        if self.type == 'rgb':
            from pycis.tools.color_system import cs_srgb
            signal = cs_srgb.spec_to_rgb(spectrum, )

        else:
            signal = spectrum.integrate(coord='wavelength')

            if not clean:
                np.random.seed()
                signal.values = np.random.poisson(signal.values)
            signal = signal * self.qe

            if not clean:
                signal.values = signal.values + np.random.normal(0, self.cam_noise, signal.values.shape)

            signal = signal / self.epercount
            signal.values = np.digitize(signal.values, np.arange(0, 2 ** self.bit_depth))
            signal = signal.astype(np.uint16)

        return signal

    def get_pixel_position(self, x_pixel=None, y_pixel=None, ):
        """
        Calculate pixel positions (in m) on the camera's sensor plane (the x-y plane).

        The origin of the x-y coordinate system is the centre of the sensor. Pixel positions correspond to the pixel
        centres.
        :param x_pixel: If specified, only return position of these x_pixels.
        :type x_pixel: float, np.array, xr.DataArray
        :param y_pixel: If specified, only return position of these y_pixels.
        :type y_pixel: float, np.array, xr.DataArray
        :return: x_pos, y_pos, both instances of xr.DataArray
        """
        centre_pos = self.pixel_size * np.array(self.sensor_format) / 2

        x = (np.arange(self.sensor_format[0]) + 0.5) * self.pixel_size - centre_pos[0]
        y = (np.arange(self.sensor_format[1]) + 0.5) * self.pixel_size - centre_pos[1]
        x = xr.DataArray(x, dims=('x', ), coords=(x, ), attrs={'units': 'm'})
        y = xr.DataArray(y, dims=('y',), coords=(y,), attrs={'units': 'm'})

        # add pixel numbers as non-dimension coordinates -- for explicit indexing and plotting
        x_pixel_coord = xr.DataArray(np.arange(self.sensor_format[0], ).astype(np.uint16), dims=('x', ), coords=(x, ), )
        y_pixel_coord = xr.DataArray(np.arange(self.sensor_format[1], ).astype(np.uint16), dims=('y', ), coords=(y, ), )
        x = x.assign_coords({'x_pixel': ('x', x_pixel_coord), }, )
        y = y.assign_coords({'y_pixel': ('y', y_pixel_coord), }, )

        if x_pixel is not None:
            x = x.isel(x=x_pixel)
        if y_pixel is not None:
            y = y.isel(y=y_pixel)

        return x, y

    def get_mueller_matrix(self, ):
        """
        :return:
        """
        idxs_1, idxs_2, idxs_3, idxs_4 = get_pixel_idxs(self.sensor_format)

        mat = np.zeros([self.sensor_format[0], self.sensor_format[1], 4, 4, ])
        dims = ('x', 'y', 'mueller_v', 'mueller_h', )
        mat = xr.DataArray(mat, dims=dims, ).assign_coords({'x': self.x, 'y': self.y, }, )

        mat[idxs_1[0], idxs_1[1], ..., ] = LinearPolariser(orientation=0).get_mueller_matrix()
        mat[idxs_2[0], idxs_2[1], ..., ] = LinearPolariser(orientation=45).get_mueller_matrix()
        mat[idxs_3[0], idxs_3[1], ..., ] = LinearPolariser(orientation=90).get_mueller_matrix()
        mat[idxs_4[0], idxs_4[1], ..., ] = LinearPolariser(orientation=135).get_mueller_matrix()

        return mat

    def get_pixelated_phase_mask(self, ):
        """
        Calls the fn. camera.calc_pixelated_phase_mask and assigns the correct x, y coordinates.

        :return:
        """
        return get_pixelated_phase_mask(self.sensor_format).assign_coords({'x': self.x, 'y': self.y, }, )

    def __eq__(self, other_cam):
        args = [getattr(self, arg) for arg in list(inspect.signature(Camera).parameters)]
        other_args = [getattr(other_cam, arg) for arg in list(inspect.signature(Camera).parameters)]
        return args == other_args


def get_pixelated_phase_mask(sensor_format):
    """
    pixelated phase mask for the standard polarised CI instrument layout described in my thesis.

    :param sensor_format: (tuple) number of pixels in each dimension (x, y, ).
    :return: (xr.DataArray) phase_mask with dimensions 'x' and 'y' and without coordinates.
    """

    phase_mask = xr.DataArray(np.zeros(sensor_format), dims=('x', 'y', ), )
    idxs_x = xr.DataArray(np.arange(0, sensor_format[0], 2), dims=('x',), )
    pix_idxs_y = xr.DataArray(np.arange(0, sensor_format[1], 2), dims=('y',), )

    phase_mask[idxs_x, pix_idxs_y, ] = 0
    phase_mask[idxs_x + 1, pix_idxs_y, ] = np.pi / 2
    phase_mask[idxs_x + 1, pix_idxs_y + 1, ] = np.pi
    phase_mask[idxs_x, pix_idxs_y + 1, ] = 3 * np.pi / 2

    return phase_mask


def get_pixel_idxs(sensor_format):
    """
    Get indices of pixel numbers, determining the pixelated polariser layout.

    Following the pixel-numbering conventions defined for the FLIR Blackfly S camera in my paper. This may need to be
    generalised at some point.

    :return: idxs1
    """

    idxs_x = xr.DataArray(np.arange(0, sensor_format[0], 2), dims=('x',), )
    idxs_y = xr.DataArray(np.arange(0, sensor_format[1], 2), dims=('y',), )

    idxs1 = idxs_x, idxs_y
    idxs2 = idxs_x + 1, idxs_y
    idxs3 = idxs_x + 1, idxs_y + 1
    idxs4 = idxs_x, idxs_y + 1

    return idxs1, idxs2, idxs3, idxs4


def get_superpixel_position(x, y, ):
    """
    given pixel positions x and y, return the 2x2 superpixel positions xs and ys

    :return:
    """

    sensor_format = [len(x), len(y)]
    pixel_size = float(x[1] - x[0])
    idxs1, _, _, _ = get_pixel_idxs(sensor_format)

    xs = x.isel(x=idxs1[0]).values + pixel_size / 2
    xs = xr.DataArray(xs, coords=(xs, ), dims=('x', ))
    xs.attrs = {'units': 'm'}

    ys = y.isel(y=idxs1[1]).values + pixel_size / 2
    ys = xr.DataArray(ys, coords=(ys, ), dims=('y', ))
    ys.attrs = {'units': 'm'}

    return xs, ys




