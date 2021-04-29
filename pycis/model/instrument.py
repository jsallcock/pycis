import sys
import os
import yaml
import numpy as np
import xarray as xr
from math import isclose, radians
from numba import vectorize, f8
import pycis
from pycis.model import mueller_product, LinearPolariser, Camera, QuarterWaveplate, Component, LinearRetarder, \
    UniaxialCrystal


class Instrument:
    """
    Coherence imaging instrument.

    Generated from either a configuration file, or with the constituent python objects.

    :param str config: path to a .yaml instrument configuration file
    :param pycis.model.Camera camera: Instrument camera.
    :param list optics: A list of floats, the focal lengths (in m) of the three lenses used in the standard CI
        configuration (see e.g. my thesis or Scott Silburn's): [f_1, f_2, f_3] where f_1 is the objective lens.
    :param list interferometer: A list of instances of pycis.model.Component, where the first entry is the first
        component that the light passes through.
    :param bool force_mueller: Forces the full Mueller matrix calculation of the interferogram, regardless of whether an
        analytical shortcut is available.
    """
    def __init__(self, config=None, camera=None, optics=None, interferometer=None, force_mueller=False):

        if config is not None:
            self.camera, self.optics, self.interferometer = self.parse_config(config)
        else:
            self.camera = camera
            self.optics = optics
            self.interferometer = interferometer

        self.force_mueller = force_mueller
        self.check_inputs()
        self.crystals = [c for c in self.interferometer if isinstance(c, LinearRetarder)]
        self.polarisers = [c for c in self.interferometer if isinstance(c, LinearPolariser)]
        self.type = self.get_type()

    def parse_config(self, config):
        """
        Try loading config as an absolute path to a .yaml file. Failing that, try it as a relative path to a .yaml file
        from the current working directory. Finally, try looking for config as a .yaml file saved in
        pycis/model/config/.
        """

        fpaths = [
            config,
            os.path.join(os.getcwd(), config),
            os.path.join(pycis.root, 'model', 'config', config),
        ]
        for fpath in fpaths:
            try:
                with open(fpath) as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                found = True
            except FileNotFoundError:
                found = False

        if not found:
            print('pycis: could not find config file')
            raise FileNotFoundError

        try:
            camera = Camera(**config['camera'])
            optics = [config['focal_length_lens_' + str(i + 1)] for i in range(3)]
            ic = config['interferometer']
            interferometer = [getattr(pycis, [*ic[i]][0])(**[*ic[i].values()][0]) for i in range(3)]

            for component in interferometer:
                component.orientation += config['interferometer_orientation']

        except:
            print('pycis: could not interpret config file')
            raise FileNotFoundError

        return camera, optics, interferometer

    def check_inputs(self):
        assert isinstance(self.camera, Camera)
        assert isinstance(self.optics, list)
        assert all(isinstance(co, Component) for co in self.interferometer)

    def get_type(self):
        """
        Type of instrument determines how interferogram is calculated.

        Valid instrument types:
        - 'mueller': Full Mueller calculation
        - 'single_delay_linear'
        - 'single_delay_pixelated'
        - 'multi_delay_pixelated'

        :return: type (str)
        """

        if self.force_mueller:
            return 'mueller'

        type = None
        if self.camera.type == 'monochrome_polarised':

            # single-delay polarised
            if len(self.interferometer) == 3:

                # check for correct component types and relative orientations
                types = [LinearPolariser, UniaxialCrystal, QuarterWaveplate, ]
                relative_orientations = [0, 45, 90, ]

                conditions_met = []
                for idx, (typ, rel_or) in enumerate(zip(types, relative_orientations)):
                    component = self.interferometer[idx]
                    conditions_met.append(isinstance(component, typ))
                    conditions_met.append(isclose(component.orientation - self.polarisers[0].orientation, rel_or))

                if all(conditions_met):
                    type = 'single_delay_pixelated'

            # multi-delay polarised
            elif len(self.interferometer) == 5:

                # check for correct component types and relative orientations
                types = [LinearPolariser, UniaxialCrystal, LinearPolariser, UniaxialCrystal, QuarterWaveplate, ]
                relative_orientations = [0, 45, 0, 45, 90, ]

                conditions_met = []
                for idx, (typ, rel_or) in enumerate(zip(types, relative_orientations)):
                    component = self.interferometer[idx]
                    conditions_met.append(isinstance(component, typ))
                    conditions_met.append(isclose(component.orientation - self.polarisers[0].orientation, rel_or))

                if all(conditions_met):
                    type = 'multi_delay_pixelated'

        # are there two polarisers, at the front and back of the interferometer?
        elif len(self.polarisers) == 2 and (isinstance(self.interferometer[0], LinearPolariser) and
                                            isinstance(self.interferometer[-1], LinearPolariser)):

            pol_1_orientation = self.interferometer[0].orientation
            pol_2_orientation = self.interferometer[-1].orientation

            conditions_met = [pol_1_orientation == pol_2_orientation, ] + \
                             [isclose(crys.orientation - pol_1_orientation, 45) for crys in self.crystals]
            if all(conditions_met):
                type = 'single_delay_linear'

        if type is None:
            type = 'mueller'

        return type

    # def get_inc_angle(self, x, y):
    #     """
    #     Calculate incidence angle(s) of ray(s) through the interferometer, in radians.
    #
    #     :param x: Pixel centre x position(s) in sensor plane in m.
    #     :type x: float, xr.DataArray
    #     :param y: Pixel centre y position(s) in sensor plane in m.
    #     :type y: float, xr.DataArray
    #     :return: (float, xr.DataArray) Incidence angle(s) in radians.
    #     """
    #     # return xr.apply_ufunc(_get_inc_angle, x, y, self.optics[2], dask='allowed', )
    #     return np.arctan2((x ** 2 + y ** 2) ** 0.5, self.optics[2], )

    def get_inc_angle(self, x, y, component):
        """
        Calculate incidence angle(s) of ray(s) through the interferometer, in radians.

        :param x: Pixel centre x position(s) in sensor plane in m.
        :type x: float, xr.DataArray
        :param y: Pixel centre y position(s) in sensor plane in m.
        :type y: float, xr.DataArray
        :param pycis.Component component: Interferometer component.
        :return: (float, xr.DataArray) Incidence angle(s) in radians.
        """

        x0 = self.optics[2] * np.tan(radians(component.tilt_x))
        y0 = self.optics[2] * np.tan(radians(component.tilt_y))

        # return xr.apply_ufunc(_get_inc_angle, x, y, self.optics[2], dask='allowed', )
        return np.arctan2(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5, self.optics[2], )

    def get_azim_angle(self, x, y, component):
        """
        Calculate azimuthal angle(s) of ray(s) through the crystal, in radians.

        :param x: Pixel centre x position(s) in sensor plane in m.
        :type x: float, xr.DataArray
        :param y: Pixel centre y position(s) in sensor plane in m.
        :type y: float, xr.DataArray
        :param pycis.OrientableComponent component: Interferometer component.

        :return: (float, xr.DataArray) Azimuthal angle(s) in radians.
        """
        return np.arctan2(y, x) + np.pi - radians(component.orientation)
        # return xr.apply_ufunc(_get_azim_angle, x, y, radians(crystal.orientation), dask='allowed', )

    def get_mueller_matrix(self, wavelength, x, y):
        """
        Calculate total Mueller matrix for the interferometer.

        :param wavelength: Wavelength in m. I
        :type wavelength: float, xarray.DataArray
        :param x: Pixel centre x position(s) in sensor plane in m.
        :type x: float, xarray.DataArray
        :param y: Pixel centre y position(s) in sensor plane in m.
        :type y: float, xarray.DataArray
        :return: (xr.DataArray) Mueller matrix.
        """

        mat_total = xr.DataArray(np.identity(4), dims=('mueller_v', 'mueller_h'), )

        for component in self.interferometer:
            inc_angle = self.get_inc_angle(x, y, component)
            azim_angle = self.get_azim_angle(x, y, component)
            mat_component = component.get_mueller_matrix(wavelength, inc_angle, azim_angle)
            mat_total = mueller_product(mat_component, mat_total)
        return mat_total

    def get_delay(self, wavelength, x, y):
        """
        Calculate the interferometer delay(s) at the given wavelength(s).

        :param wavelength: Wavelength in m. If xarray.DataArray, must have dimension name 'wavelength'.
        :type wavelength: float, xarray.DataArray
        :param x: Pixel centre x position(s) in sensor plane in m.
        :type x: float, xarray.DataArray
        :param y: Pixel centre y position(s) in sensor plane in m.
        :type y: float, xarray.DataArray
        :return: (xr.DataArray) Interferometer delay(s) in radians.
        """

        # not sure it would be possible to write a general function for this
        assert self.type != 'mueller'

        inc_angle = self.get_inc_angle(x, y, self.crystals[0])
        azim_angle = self.get_azim_angle(x, y, self.crystals[0])

        # calculation depends on instrument type
        if self.type == 'single_delay_linear':
            # add delay contribution due to each crystal
            delay = 0
            for crystal in self.crystals:
                delay += crystal.get_delay(wavelength, inc_angle, azim_angle, )

        elif self.type == 'single_delay_pixelated':
            # generalise to arbitrary interferometer orientations
            orientation_delay = -2 * radians(self.polarisers[0].orientation)
            delay = self.crystals[0].get_delay(wavelength, inc_angle, azim_angle, ) + orientation_delay

        elif self.type == 'multi_delay_pixelated':
            # generalise to arbitrary interferometer orientations
            orientation_delay = -2 * radians(self.polarisers[0].orientation)
            delay_1 = self.crystals[0].get_delay(wavelength, inc_angle, azim_angle, )
            delay_2 = self.crystals[1].get_delay(wavelength, inc_angle, azim_angle, ) + orientation_delay
            delay_sum = delay_1 + delay_2
            delay_diff = abs(delay_1 - delay_2)

            delay = delay_1, delay_2, delay_sum, delay_diff
        else:
            raise NotImplementedError

        return delay

    def capture(self, spectrum, clean=False):
        """
        Capture image of given spectrum.

        :param spectrum: (xarray.DataArray) photon fluence spectrum with units of ph / m [hitting the pixel area during
            exposure time] and with dimensions 'wavelength' and, optionally, 'x', 'y' and 'stokes'. Xarray broadcasting
            rules apply to the spatial dimensions: if 'x' or 'y' are not in spectrum.dims then it is assumed that the
            incident spectrum is uniform across pixels. However, if there is no 'stokes' dimension then it is assumed
            that light is unpolarised (i.e. the spectrum supplied is the S_0 Stokes parameter only).
        :param bool clean: False to add realistic image noise, passed to self.camera.capture()
        :return: (xarray.DataArray) image in units of camera counts.
        """

        if 'x' in spectrum.dims:
            x = spectrum.x
        else:
            x = self.camera.x

        if 'y' in spectrum.dims:
            y = spectrum.y
        else:
            y = self.camera.y

        if self.type == 'mueller':
            # do the full Mueller matrix calculation

            if 'stokes' not in spectrum.dims:
                a0 = xr.zeros_like(spectrum)
                spectrum = xr.combine_nested([spectrum, a0, a0, a0], concat_dim=('stokes',))

            mueller_matrix_total = self.get_mueller_matrix(spectrum.wavelength, x, y)
            spectrum = mueller_product(mueller_matrix_total, spectrum)
            apply_polarisers = None

        else:
            delay = self.get_delay(spectrum.wavelength, x, y)
            apply_polarisers = False

            if self.type == 'single_delay_linear' and 'stokes' not in spectrum.dims:
                contrast = np.array([crystal.contrast for crystal in self.crystals]).prod()
                spectrum = spectrum / 4 * (1 + contrast * np.cos(delay))

            elif self.type == 'single_delay_pixelated' and 'stokes' not in spectrum.dims:
                contrast = self.crystals[0].contrast
                phase_mask = self.camera.get_pixelated_phase_mask()
                spectrum = spectrum / 4 * (1 + contrast * np.cos(delay + phase_mask))

            elif self.type == 'multi_delay_pixelated' and 'stokes' not in spectrum.dims:

                phase_mask = self.camera.get_pixelated_phase_mask()

                delay_1, delay_2, delay_sum, delay_diff = delay
                contrast_1 = self.crystals[0].contrast
                contrast_2 = self.crystals[0].contrast
                contrast_sum = contrast_diff = contrast_1 * contrast_2

                spectrum = spectrum / 8 * (1 +
                                           contrast_1 * np.cos(delay_1) +
                                           contrast_2 * np.cos(delay_2 + phase_mask) +
                                           1 / 2 * contrast_sum * np.cos(delay_sum + phase_mask) +
                                           1 / 2 * contrast_diff * np.cos(delay_diff + phase_mask)
                                           )
            else:
                raise NotImplementedError

        image = self.camera.capture(spectrum, apply_polarisers=apply_polarisers, clean=clean)
        return image

    def get_fringe_frequency(self, wavelength):
        """
        Calculate the interference fringe frequency at the sensor plane for the given wavelength.

        :param float wavelength: Wavelength in m.
        :return: (tuple) x and y components of the fringe frequency in units m^-1 and in order (f_x, f_y).
        """
        assert self.type != 'mueller'

        if self.type == 'single_delay_linear':
            # add contribution due to each crystal
            spatial_freq_x, spatial_freq_y = 0, 0
            for crystal in self.crystals:

                sp_f_x, sp_f_y = crystal.get_fringe_frequency(wavelength, self.optics[2], )
                spatial_freq_x += sp_f_x
                spatial_freq_y += sp_f_y

        elif self.type == 'multi_delay_pixelated':
            crystal = self.crystals[0]
            spatial_freq_x, spatial_freq_y = crystal.get_fringe_frequency(wavelength, self.optics[2], )
            # TODO and also the sum and difference terms?

        else:
            raise NotImplementedError

        return spatial_freq_x, spatial_freq_y



# @vectorize([f8(f8, f8, f8, ), ], nopython=True, fastmath=False, cache=False, )
# def _get_inc_angle(x, y, f_3):
#     return np.arctan2((x ** 2 + y ** 2) ** 0.5, f_3, )


# @vectorize([f8(f8, f8, f8, ), ], nopython=True, fastmath=False, cache=False, )
# def _get_azim_angle(x, y, crystal_orientation, ):
#     return np.arctan2(y, x) + np.pi - crystal_orientation
