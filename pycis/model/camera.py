import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pycis.model import LinearPolariser, mueller_product


class Camera(object):

    def __init__(self, bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise, polarised=False):
        """

        :param bit_depth: 
        :param sensor_format: (x, y, )
        :param pix_size: pixel dimension [ m ].
        :param qe: Quantum efficiency or sensor.
        :param epercount: Conversion gain of sensor.
        :param cam_noise: Camera noise standard deviation [ e- ].

        """

        self.pixel_size = pixel_size
        self.sensor_format = sensor_format
        self.qe = qe
        self.epercount = epercount
        self.cam_noise = cam_noise
        self.bit_depth = bit_depth
        self.polarised = polarised
        self.x, self.y = self.calculate_pixel_position()

        if polarised:
            assert sensor_format[0] % 2 == 0
            assert sensor_format[1] % 2 == 0

    def capture_image(self, image, apply_polarisers=None):
        """
        capture image of scene

        :param image: (xr.DataArray) image in units of photons with dimensions 'x', 'y' and (optionally) 'stokes'. If
        no stokes dim then it is assumed that light is unpolarised (i.e. the spec supplied is the S_0 Stokes parameter)
        :return:
        """

        if apply_polarisers is None:
            apply_polarisers = self.polarised

        if apply_polarisers:
            assert 'stokes' in image.dims
            mueller_matrix = self.calculate_mueller_matrix()
            image = mueller_product(mueller_matrix, image, )

        if 'stokes' in image.dims:
            image = image.isel(stokes=0, drop=True)

        if 'wavelength' in image.dims:
            if len(image.wavelength) >= 2:
                signal = image.integrate(dim='wavelength')
        else:
            signal = image

        np.random.seed()
        signal.values = np.random.poisson(signal.values)
        signal = signal * self.qe
        signal.values = signal.values + np.random.normal(0, self.cam_noise, signal.values.shape)
        signal = signal / self.epercount
        signal.values = np.digitize(signal.values, np.arange(0, 2 ** self.bit_depth))

        return signal

    def calculate_pixel_position(self, x_pixel=None, y_pixel=None, ):
        """
        Calculate pixel positions (in m) on the camera's sensor plane (the x-y plane).

        The origin of the x-y coordinate system is the centre of the sensor. Pixel positions correspond to the pixel
        centres. If x_pixel and y_pixel are specified then only returns the position of that pixel.

        :param x_pixel:
        :param y_pixel:
        :return: x_pos, y_pos, both instances of xr.DataArray
        """

        centre_pos = self.pixel_size * np.array(self.sensor_format) / 2

        x = (np.arange(self.sensor_format[0]) + 0.5) * self.pixel_size - centre_pos[0]
        y = (np.arange(self.sensor_format[1]) + 0.5) * self.pixel_size - centre_pos[1]
        x = xr.DataArray(x, dims=('x', ), coords=(x, ), )
        y = xr.DataArray(y, dims=('y',), coords=(y,), )

        # add pixel numbers as non-dimension coordinates -- for explicit indexing and plotting
        x_pixel_coord = xr.DataArray(np.arange(self.sensor_format[0], ), dims=('x', ), coords=(x, ), )
        y_pixel_coord = xr.DataArray(np.arange(self.sensor_format[1], ), dims=('y', ), coords=(y, ), )
        x = x.assign_coords({'x_pixel': ('x', x_pixel_coord), }, )
        y = y.assign_coords({'y_pixel': ('y', y_pixel_coord), }, )

        if x_pixel is not None:
            x = x.isel(x=x_pixel)
        if y_pixel is not None:
            y = y.isel(y=y_pixel)

        return x, y

    def calculate_superpixel_position(self):
        """

        :return:
        """

        sensor_format_s = np.array(self.sensor_format) / 2
        pixel_size_s = self.pixel_size * 2
        centre_pos = pixel_size_s * sensor_format_s / 2

        x = (np.arange(sensor_format_s[0]) + 0.5) * pixel_size_s - centre_pos[0]
        y = (np.arange(sensor_format_s[1]) + 0.5) * pixel_size_s - centre_pos[1]
        x = xr.DataArray(x, dims=('x',), coords=(x,), )
        y = xr.DataArray(y, dims=('y',), coords=(y,), )

        # add superpixel numbers as non-dimension coordinates -- for explicit indexing and plotting
        x_superpixel_coord = xr.DataArray(np.arange(self.sensor_format[0], ), dims=('x',), coords=(x,), )
        y_superpixel_coord = xr.DataArray(np.arange(self.sensor_format[1], ), dims=('y',), coords=(y,), )
        x = x.assign_coords({'x_superpixel': ('x', x_superpixel_coord), }, )
        y = y.assign_coords({'y_superpixel': ('y', y_superpixel_coord), }, )

        return x, y

    def calculate_mueller_matrix(self, ):
        """

        :return:
        """

        args = [None, ] * 3  # ok this is grim
        matrix_0deg = LinearPolariser(0).calculate_matrix(*args)
        matrix_45deg = LinearPolariser(np.pi / 4).calculate_matrix(*args)
        matrix_90deg = LinearPolariser(np.pi / 2).calculate_matrix(*args)
        matrix_135deg = LinearPolariser(3 * np.pi / 4).calculate_matrix(*args)

        pix_idxs_x = xr.DataArray(np.arange(0, self.sensor_format[0], 2), dims=('x',), )
        pix_idxs_y = xr.DataArray(np.arange(0, self.sensor_format[1], 2), dims=('y', ), )

        mueller_matrix = np.zeros([self.sensor_format[0], self.sensor_format[1], 4, 4, ])
        dims = ('x', 'y', 'mueller_v', 'mueller_h', )
        mueller_matrix = xr.DataArray(mueller_matrix, dims=dims, ).assign_coords({'x': self.x, 'y': self.y, }, )

        mueller_matrix[pix_idxs_x, pix_idxs_y, ...] = matrix_0deg
        mueller_matrix[pix_idxs_x + 1, pix_idxs_y, ..., ] = matrix_45deg
        mueller_matrix[pix_idxs_x + 1, pix_idxs_y + 1, ..., ] = matrix_90deg
        mueller_matrix[pix_idxs_x, pix_idxs_y + 1, ..., ] = matrix_135deg

        return mueller_matrix

    def calculate_pixelated_phase_mask(self, ):
        """

        :return:
        """

        pix_idxs_x = xr.DataArray(np.arange(0, self.sensor_format[0], 2), dims=('x',), )
        pix_idxs_y = xr.DataArray(np.arange(0, self.sensor_format[1], 2), dims=('y',), )
        phase_mask = np.zeros(self.sensor_format, )
        dims = ('x', 'y',)
        phase_mask = xr.DataArray(phase_mask, dims=dims, ).assign_coords({'x': self.x, 'y': self.y, }, )

        phase_mask[pix_idxs_x, pix_idxs_y, ] = 0
        phase_mask[pix_idxs_x + 1, pix_idxs_y, ] = np.pi / 2
        phase_mask[pix_idxs_x + 1, pix_idxs_y + 1, ] = np.pi
        phase_mask[pix_idxs_x, pix_idxs_y + 1, ] = 3 * np.pi / 2

        return phase_mask