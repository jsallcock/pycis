import numpy as np
import xarray as xr
from pycis.model import LinearPolariser, mueller_product

camera_modes = ['mono',
                'mono_polarised',
                'rgb',
                ]


class Camera(object):

    def __init__(self, bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise, mode='mono'):
        """

        :param bit_depth: 
        :param sensor_format: (x, y, )
        :param pix_size: pixel dimension [ m ].
        :param qe: Quantum efficiency or sensor.
        :param epercount: Conversion gain of sensor.
        :param cam_noise: Camera noise standard deviation [ e- ].
        :param mode:

        """

        self.pixel_size = pixel_size
        self.sensor_format = sensor_format
        self.qe = qe
        self.epercount = epercount
        self.cam_noise = cam_noise
        self.bit_depth = bit_depth
        self.mode = mode
        self.x, self.y = self.calculate_pixel_position()

        assert mode in camera_modes
        if mode == 'mono_polarised':
            assert sensor_format[0] % 2 == 0
            assert sensor_format[1] % 2 == 0

    def capture(self, spectrum, apply_polarisers=None, clean=False):
        """
        Capture an image

        :param spectrum: (xr.DataArray) in units of photons / m with dimensions 'x', 'y', 'wavelength' and (optionally)
        'stokes'. If there is no stokes dim then the light is assumed to be unpolarised.
        :param apply_polarisers: (bool) whether to apply the camera's pixelated polariser array to the spectrum.
        :param clean: (bool) False to add realistic image noise. Clean images used for testing.
        :return:
        """

        if apply_polarisers is None:
            if self.mode == 'mono_polarised':
                apply_polarisers = True

        if apply_polarisers:
            assert 'stokes' in spectrum.dims
            mueller_matrix = self.calculate_mueller_matrix()
            spectrum = mueller_product(mueller_matrix, spectrum, )

        # ensure only total intensity (first Stokes parameter) is observed
        if 'stokes' in spectrum.dims:
            spectrum = spectrum.isel(stokes=0, drop=True)

        if self.mode == 'rgb':
            from pycis.tools.color_system import cs_srgb
            signal = cs_srgb.spec_to_rgb(spectrum, )

        else:
            signal = spectrum.integrate(dim='wavelength')

            if not clean:
                np.random.seed()
                signal.values = np.random.poisson(signal.values)
            signal = signal * self.qe

            if not clean:
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

        pix_idxs_x = xr.DataArray(np.arange(0, self.sensor_format[0], 2), dims=('x', ), )
        pix_idxs_y = xr.DataArray(np.arange(0, self.sensor_format[1], 2), dims=('y', ), )

        mat = np.zeros([self.sensor_format[0], self.sensor_format[1], 4, 4, ])
        dims = ('x', 'y', 'mueller_v', 'mueller_h', )
        mat = xr.DataArray(mat, dims=dims, ).assign_coords({'x': self.x, 'y': self.y, }, )

        mat[pix_idxs_x, pix_idxs_y, ..., ] = LinearPolariser(0).calculate_mueller_matrix()
        mat[pix_idxs_x + 1, pix_idxs_y, ..., ] = LinearPolariser(np.pi / 4).calculate_mueller_matrix()
        mat[pix_idxs_x + 1, pix_idxs_y + 1, ..., ] = LinearPolariser(np.pi / 2).calculate_mueller_matrix()
        mat[pix_idxs_x, pix_idxs_y + 1, ..., ] = LinearPolariser(3 * np.pi / 4).calculate_mueller_matrix()

        return mat

    def calculate_pixelated_phase_mask(self, ):
        """

        :return:
        """

        pix_idxs_x = xr.DataArray(np.arange(0, self.sensor_format[0], 2), dims=('x', ), )
        pix_idxs_y = xr.DataArray(np.arange(0, self.sensor_format[1], 2), dims=('y', ), )
        phase_mask = np.zeros(self.sensor_format, )
        dims = ('x', 'y',)
        phase_mask = xr.DataArray(phase_mask, dims=dims, ).assign_coords({'x': self.x, 'y': self.y, }, )

        phase_mask[pix_idxs_x, pix_idxs_y, ] = 0
        phase_mask[pix_idxs_x + 1, pix_idxs_y, ] = np.pi / 2
        phase_mask[pix_idxs_x + 1, pix_idxs_y + 1, ] = np.pi
        phase_mask[pix_idxs_x, pix_idxs_y + 1, ] = 3 * np.pi / 2

        return phase_mask
