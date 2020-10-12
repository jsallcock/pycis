import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pycis.model import LinearPolariser, mueller_product


class Camera(object):
    """
    Camera base class

    """

    def __init__(self, bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise):
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

    def calculate_pixel_position(self, x_pixel=None, y_pixel=None, crop=None, downsample=1):
        """
        Calculate pixel positions (in m) on the camera's sensor plane (the x-y plane).

        The origin of the x-y coordinate system is the centre of the sensor. Pixel positions correspond to the pixel
        centres. If x_pixel and y_pixel are specified then only returns the position of that pixel. crop and downsample
        are essentially legacy kwargs from my thesis work.

        :param x_pixel:
        :param y_pixel:
        :param crop: (y1, y2, x1, x2)
        :param downsample:
        :return: x_pos, y_pos, both instances of xr.DataArray
        """

        centre_pos = self.pixel_size * np.array(self.sensor_format) / 2  # relative to x=0, y=0 pixel

        if crop is None:
            crop = (0, self.sensor_format[0], 0, self.sensor_format[1])

        x_coord = np.arange(crop[0], crop[1])[::downsample]
        y_coord = np.arange(crop[2], crop[3])[::downsample]
        x = (x_coord + 0.5) * self.pixel_size - centre_pos[0]
        y = (y_coord + 0.5) * self.pixel_size - centre_pos[1]
        x = xr.DataArray(x, dims=('x', ), coords=(x, ), )
        y = xr.DataArray(y, dims=('y',), coords=(y,), )

        # add pixel numbers as non-dimension coordinates -- just for explicit indexing and plotting
        x_pixel_coord = xr.DataArray(np.arange(self.sensor_format[0], ), dims=('x', ), coords=(x, ), )
        y_pixel_coord = xr.DataArray(np.arange(self.sensor_format[1], ), dims=('y', ), coords=(y, ), )
        x = x.assign_coords({'x_pixel': ('x', x_pixel_coord), }, )
        y = y.assign_coords({'y_pixel': ('y', y_pixel_coord), }, )

        if x_pixel is not None:
            x = x.isel(x=x_pixel)
        if y_pixel is not None:
            y = y.isel(y=y_pixel)

        return x, y

    def capture_image(self, spec, ):
        """

        capture image of scene

        :param spec: (xr.DataArray) input spectrum with dimensions 'wavelength', 'x', 'y' and (optionally) 'stokes'. If
        no stokes dim then it is assumed that light is unpolarised (i.e. the spec supplied is the S_0 Stokes parameter)
        :return:
        """

        np.random.seed()
        if 'stokes' in spec.dims:
            spec = spec.isel(stokes=0, drop=True)

        if 'wavelength' in spec.dims:
            if len(spec.wavelength) >= 2:
                signal = spec.integrate(dim='wavelength')
        else:
            signal = spec

        signal = signal * self.qe
        # signal.values = np.random.poisson(signal.values)
        # signal.values = signal.values + np.random.normal(0, self.cam_noise, signal.values.shape)
        signal = signal / self.epercount
        # signal.values = np.digitize(signal.values, np.arange(0, 2 ** self.bit_depth))

        return signal


class PolarisedCamera(Camera):
    """
    Camera with a pixelated polariser array on its sensor e.g. FLIR Blackfly S, Photron Chrysta
    """

    def __init__(self, bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise):
        """

        :param format:
        :type format: str

        """

        assert sensor_format[0] % 2 == 0
        assert sensor_format[1] % 2 == 0

        super().__init__(bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise)

        # define Mueller matrix
        args = [None, ] * 3  # ok this is grim
        matrix_0deg = LinearPolariser(0).calculate_matrix(*args)
        matrix_45deg = LinearPolariser(np.pi / 4).calculate_matrix(*args)
        matrix_90deg = LinearPolariser(np.pi / 2).calculate_matrix(*args)
        matrix_135deg = LinearPolariser(3 * np.pi / 4).calculate_matrix(*args)

        x, y, = self.calculate_pixel_position()
        pix_idxs_x = xr.DataArray(np.arange(0, self.sensor_format[0], 2), dims=('x',), )
        pix_idxs_y = xr.DataArray(np.arange(0, self.sensor_format[1], 2), dims=('y', ), )

        mueller_matrix = np.zeros([self.sensor_format[0], self.sensor_format[1], 4, 4, ])
        dims = ('x', 'y', 'mueller_v', 'mueller_h', )
        mueller_matrix = xr.DataArray(mueller_matrix, dims=dims, ).assign_coords({'x': x, 'y': y, }, )
        mueller_matrix[pix_idxs_x, pix_idxs_y, ...] = matrix_0deg
        mueller_matrix[pix_idxs_x + 1, pix_idxs_y, ..., ] = matrix_45deg
        mueller_matrix[pix_idxs_x + 1, pix_idxs_y + 1, ..., ] = matrix_90deg
        mueller_matrix[pix_idxs_x, pix_idxs_y + 1, ..., ] = matrix_135deg

        self.mueller_matrix = mueller_matrix

    def capture_image(self, spec, ):
        """

        :param intensity:
        :param clean:
        :param display:
        :return:
        """

        assert 'stokes' in spec.dims
        spec = mueller_product(self.mueller_matrix, spec, )
        spec = spec.isel(stokes=0, drop=True)

        if 'wavelength' in spec.dims:
            if len(spec.wavelength) >= 2:
                signal = spec.integrate(dim='wavelength')
        else:
            signal = spec

        signal = signal * self.qe
        # np.random.seed()
        # signal.values = np.random.poisson(signal.values)
        # signal.values = signal.values + np.random.normal(0, self.cam_noise, signal.values.shape)
        signal = signal / self.epercount
        # signal.values = np.digitize(signal.values, np.arange(0, 2 ** self.bit_depth))

        return signal


## LEGACY METHOD ##
# def capture_stack(self, photon_fluence, num_stack, display=False):
#     """ Quickly capture of a stack of image frames, returning the total signal. """
#
#     # older implementation loops over Camera.capture() method and is far, far slower:
#     # pool = mp.Pool(processes=2)
#     # total_signal = sum(pool.map(self.capture, [photon_fluence] * num_stack))
#
#     stacked_photon_fluence = num_stack * photon_fluence
#
#     electron_fluence = photon_fluence * self.qe
#     stacked_electron_fluence = stacked_photon_fluence * self.qe
#
#     electron_noise_std = np.sqrt(electron_fluence + self.cam_noise ** 2)
#     stacked_electron_noise_std = np.sqrt(num_stack * (electron_noise_std))
#
#     stacked_electron_fluence += np.random.normal(0, stacked_electron_noise_std, self.sensor_dim)
#
#     # apply gain
#     # signal = electron_fluence / self.epercount
#     stacked_signal = stacked_electron_fluence / self.epercount
#
#     # digitise at bitrate of sensor
#     # signal = np.digitize(signal, np.arange(2 ** self.bit_depth))
#     stacked_signal = np.digitize(stacked_signal, np.arange(num_stack * 2 ** self.bit_depth))
#
#     if display:
#         plt.figure()
#         plt.imshow(stacked_signal, 'gray')
#         plt.colorbar()
#         plt.show()
#
#     return stacked_signal

if __name__ == '__main__':
    polcam = PolarisedCamera(12, (2500, 2000), 3.45e-6, 0.45, 1, 1, )
