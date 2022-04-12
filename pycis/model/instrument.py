import copy
import sys
import os
import inspect
import yaml
from datetime import datetime
import numpy as np
import xarray as xr
from math import isclose, radians, prod
from numba import vectorize, f8
from fnmatch import fnmatch
import pycis
from pycis.model import mueller_product, LinearPolariser, Camera, QuarterWaveplate, Component, LinearRetarder, \
    UniaxialCrystal, TiltableComponent


class Instrument:
    """
    Coherence imaging instrument

    :param str config: \
        Path to a .yaml instrument configuration file.

    :param pycis.model.Camera camera: \
        Instrument camera.

    :param list optics: \
        A list of floats, the focal lengths (in m) of the three lenses used in the standard CI configuration (see e.g.
        my thesis or Scott Silburn's): [f_1, f_2, f_3] where f_1 is the objective lens.

    :param list interferometer: \
        A list of instances of pycis.model.Component, where the first entry is the first
        component that the light passes through.

    :param bool force_mueller: \
        Forces the full Mueller matrix calculation of the interferogram, regardless of whether an
        analytical shortcut is available.

    """
    def __init__(self, config=None, camera=None, optics=None, interferometer=None, force_mueller=False):

        if config is not None:
            self.camera, self.optics, self.interferometer = self.read_config(config)
        else:
            self.camera = camera
            self.optics = optics
            self.interferometer = interferometer

        self.force_mueller = force_mueller
        self.check_inputs()
        self.retarders = [c for c in self.interferometer if isinstance(c, LinearRetarder)]
        self.polarisers = [c for c in self.interferometer if isinstance(c, LinearPolariser)]
        self.type = self.get_type()

    def read_config(self, config):
        """
        Tries loading config as an absolute path to a .yaml file. Failing that, try it as a relative path to a .yaml file
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
                break
            except FileNotFoundError:
                found = False

        if not found:
            raise FileNotFoundError('pycis: could not find config file')

        try:
            camera = Camera(**config['camera'])
            optics = [config['lens_' + str(i + 1) + '_focal_length'] for i in range(3)]
            ic = config['interferometer']
            interferometer = [getattr(pycis, [*ic[i]][0])(**[*ic[i].values()][0]) for i in range(len(ic))]

        except:
            raise ValueError('pycis: could not interpret config file')

        return camera, optics, interferometer

    def write_config(self, fpath):
        """
        Write the current instrument config to a .yaml config file that can then be reloaded at a later date.

        :param str filepath:
        """

        config = {
            'camera': dict([(arg, getattr(self.camera, arg)) for arg in list(inspect.signature(pycis.Camera).parameters)]),
            'lens_1_focal_length': self.optics[0],
            'lens_2_focal_length': self.optics[1],
            'lens_3_focal_length': self.optics[2],
            'interferometer': [dict([(type(c).__name__, vars(c))]) for c in self.interferometer]
        }

        assert fpath[-5:] == '.yaml'
        with open(fpath, 'w') as f:
            f.write('# This file was generated automatically at ' + datetime.now().strftime("%H:%M:%S, %m/%d/%Y") + '\n')
            documents = yaml.dump(config, f, Dumper=pycis.tools.MyDumper)

    def check_inputs(self):
        assert isinstance(self.camera, Camera)
        assert isinstance(self.optics, list)
        assert all(isinstance(co, Component) for co in self.interferometer)

    def get_type(self):
        """
        Instrument type determines how the interferogram is calculated

        Valid instrument types:
        - 'mueller': Full Mueller calculation
        - 'single_delay_linear'
        - 'single_delay_pixelated'
        - 'double_delay_pixelated'
        - 'triple_delay_pixelated'
        - 'double_delay_linear'
        - 'triple_delay_linear'
        - 'quad_delay_linear'

        :return: type (str)
        """

        if self.force_mueller:
            return 'mueller'

        inst_type = None
        if self.camera.type == 'monochrome_polarised':

            if len(self.interferometer) == 3:
                types = [LinearPolariser, UniaxialCrystal, QuarterWaveplate, ]
                if self._check_interferometer_config(types, [0, 45, 90, ]):
                    inst_type = 'single_delay_pixelated'

            elif len(self.interferometer) == 4:
                types = [LinearPolariser, UniaxialCrystal, UniaxialCrystal, QuarterWaveplate, ]

                if self._check_interferometer_config(types, [0, -22.5, 22.5, 67.5]):
                    inst_type = 'triple_delay_pixelated'

                elif self._check_interferometer_config(types, [0, -45, 0, 45]):
                    inst_type = 'double_delay_pixelated'

        elif self.camera.type == 'monochrome':

            if len(self.interferometer) == 4:
                types = [LinearPolariser, UniaxialCrystal, UniaxialCrystal, LinearPolariser, ]

                if self._check_interferometer_config(types, [0, -45, 0, -45]):
                    inst_type = 'double_delay_linear'

                elif self._check_interferometer_config(types, [0, -22.5, 22.5, -22.5]):
                    inst_type = 'triple_delay_linear'

                elif self._check_interferometer_config(types, [0, -22.5, 22.5, 0]):
                    inst_type = 'quad_delay_linear'

            # are there two polarisers, at the front and back of the interferometer?
            elif len(self.polarisers) == 2 and (isinstance(self.interferometer[0], LinearPolariser) and
                                                isinstance(self.interferometer[-1], LinearPolariser)):

                pol_1_orientation = self.interferometer[0].orientation
                pol_2_orientation = self.interferometer[-1].orientation

                conditions_met = [pol_1_orientation == pol_2_orientation, ] + \
                                 [isclose(crys.orientation - pol_1_orientation, 45) for crys in self.retarders]
                if all(conditions_met):
                    inst_type = 'single_delay_linear'

        if inst_type is None:
            inst_type = 'mueller'

        return inst_type

    def get_inc_angle(self, x, y, component):
        """
        Calculate incidence angle(s) of ray(s) through the component

        :param x: Pixel centre x position(s) in sensor plane in m.
        :type x: float, xr.DataArray
        :param y: Pixel centre y position(s) in sensor plane in m.
        :type y: float, xr.DataArray
        :param pycis.Component component: Interferometer component.
        :return: (float, xr.DataArray) Incidence angle(s) in radians.
        """
        if isinstance(component, TiltableComponent):
            x0 = self.optics[2] * np.tan(radians(component.tilt_x))
            y0 = self.optics[2] * np.tan(radians(component.tilt_y))
        else:
            x0 = 0
            y0 = 0
        return np.arctan2(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5, self.optics[2], )

    def get_azim_angle(self, x, y, component):
        """
        Calculate azimuthal angle(s) of ray(s) through the component

        :param x: Pixel centre x position(s) in sensor plane in m.
        :type x: float, xr.DataArray
        :param y: Pixel centre y position(s) in sensor plane in m.
        :type y: float, xr.DataArray
        :param pycis.OrientableComponent component: Interferometer component.

        :return: (float, xr.DataArray) Azimuthal angle(s) in radians.
        """
        if isinstance(component, pycis.TiltableComponent):
            x0 = self.optics[2] * np.tan(radians(component.tilt_x))
            y0 = self.optics[2] * np.tan(radians(component.tilt_y))
        else:
            x0 = 0
            y0 = 0
        return np.arctan2(y - y0, x - x0) + np.pi - radians(component.orientation)

    def get_mueller_matrix(self, wavelength, x, y):
        """
        Calculate total Mueller matrix for the interferometer

        :param wavelength: Wavelength in m.
        :type wavelength: float, xr.DataArray

        :param x: Pixel centre x position(s) in sensor plane in m.
        :type x: float, xr.DataArray

        :param y: Pixel centre y position(s) in sensor plane in m.
        :type y: float, xr.DataArray

        :return: (xr.DataArray) Mueller matrix.
        """

        mat_total = xr.DataArray(np.identity(4), dims=('mueller_v', 'mueller_h'), )
        for component in self.interferometer:
            inc_angle = self.get_inc_angle(x, y, component)
            azim_angle = self.get_azim_angle(x, y, component)
            mat_component = component.get_mueller_matrix(wavelength, inc_angle, azim_angle)
            mat_total = mueller_product(mat_component, mat_total)
        return mat_total

    def get_delay(self, wavelength, x, y, ):
        """
        Calculate the interferometer delay(s) at the given wavelength(s)

        :param wavelength: Wavelength in m. If xr.DataArray, must have dimension name 'wavelength'.
        :type wavelength: float, xr.DataArray
        :param x: Pixel centre x position(s) in sensor plane in m.
        :type x: float, xr.DataArray
        :param y: Pixel centre y position(s) in sensor plane in m.
        :type y: float, xr.DataArray
        :return: (xr.DataArray) Interferometer delay(s) in radians.
        """
        # Would be nice to write a generalised method if possible
        assert self.type != 'mueller'

        # get delay for each retarder
        delay = []
        for ret in self.retarders:
            inc_angle = self.get_inc_angle(x, y, ret)
            azim_angle = self.get_azim_angle(x, y, ret)
            delay.append(ret.get_delay(wavelength, inc_angle, azim_angle))

        # calculation depends on instrument type
        if self.type == 'single_delay_linear':
            delay_out = sum(delay)

        elif self.type == 'double_delay_linear':
            delay_sum = delay[0] + delay[1]
            delay_diff = abs(delay[0] - delay[1])
            delay_out = delay_sum, delay_diff

        elif self.type == 'triple_delay_linear':
            delay_sum = delay[0] + delay[1]
            delay_diff = abs(delay[0] - delay[1])
            delay_out = delay[1], delay_sum, delay_diff

        elif self.type == 'quad_delay_linear':
            delay_sum = delay[0] + delay[1]
            delay_diff = abs(delay[0] - delay[1])
            delay_out = delay[0], delay[1], delay_sum, delay_diff

        elif self.type == 'single_delay_pixelated':
            orientation_delay = -2 * radians(self.polarisers[0].orientation)
            delay_out = delay[0] + orientation_delay

        elif self.type == 'double_delay_pixelated':
            orientation_delay = -2 * radians(self.polarisers[0].orientation - 45)
            delay_sum = delay[0] + delay[1]
            delay_diff = abs(delay[0] - delay[1])
            delay_out = delay_sum + orientation_delay, delay_diff + orientation_delay

        elif self.type == 'triple_delay_pixelated':
            orientation_delay = -2 * radians(self.polarisers[0].orientation - 22.5)
            delay_sum = delay[0] + delay[1]
            delay_diff = abs(delay[0] - delay[1])
            delay_out = delay[1] + orientation_delay, delay_sum + orientation_delay, delay_diff + orientation_delay
        else:
            raise NotImplementedError

        return delay_out

    def capture(self, spectrum, clean=False):
        """
        Capture image of given spectrum.

        :param spectrum: (xr.DataArray) photon fluence spectrum with units of ph / m [hitting the pixel area during
            exposure time] and with dimensions 'wavelength' and, optionally, 'x', 'y' and 'stokes'. Xarray broadcasting
            rules apply to the spatial dimensions: if 'x' or 'y' are not in spectrum.dims then it is assumed that the
            incident spectrum is uniform across pixels. However, if there is no 'stokes' dimension then it is assumed
            that light is unpolarised (i.e. the spectrum supplied is the S_0 Stokes parameter only).
        :param bool clean: False to add realistic image noise, passed to self.camera.capture()
        :return: (xr.DataArray) image in units of camera counts.
        """
        if 'x' in spectrum.dims:
            x = spectrum.x
        else:
            x = self.camera.x
        if 'y' in spectrum.dims:
            y = spectrum.y
        else:
            y = self.camera.y

        failed = False
        if self.type != 'mueller':
            try:
                delay = self.get_delay(spectrum.wavelength, x, y)
                apply_polarisers = False
                phase_mask = self.camera.get_pixelated_phase_mask()
                contrast_inst = [ret.contrast_inst for ret in self.retarders]

                if 'stokes' not in spectrum.dims:
                    if self.type == 'single_delay_linear':
                        spectrum = spectrum / 4 * (1 + prod(contrast_inst) * np.cos(delay))

                    elif self.type == 'double_delay_linear':
                        contrast_inst_sum = contrast_inst_diff = prod(contrast_inst)
                        delay_sum, delay_diff = delay
                        spectrum = spectrum / 4 * (
                                1
                                + 0.5 * contrast_inst_diff * np.cos(delay_diff)
                                - 0.5 * contrast_inst_sum * np.cos(delay_sum)
                        )

                    elif self.type == 'triple_delay_linear':
                        contrast_inst_sum = contrast_inst_diff = prod(contrast_inst)
                        delay_2, delay_sum, delay_diff = delay
                        root2 = np.sqrt(2)
                        spectrum = spectrum / 4 * (
                                1
                                + root2 / 2 * contrast_inst[1] * np.cos(delay_2)
                                + root2 / 4 * contrast_inst_diff * np.cos(delay_diff)
                                - root2 / 4 * contrast_inst_sum * np.cos(delay_sum)
                        )

                    elif self.type == 'quad_delay_linear':
                        contrast_inst_sum = contrast_inst_diff = prod(contrast_inst)
                        delay_1, delay_2, delay_sum, delay_diff = delay
                        spectrum = spectrum / 4 * (
                                1
                                + 0.5 * contrast_inst[0] * np.cos(delay_1)
                                + 0.5 * contrast_inst[1] * np.cos(delay_2)
                                + 0.25 * contrast_inst_diff * np.cos(delay_diff)
                                - 0.25 * contrast_inst_sum * np.cos(delay_sum)
                        )

                    elif self.type == 'single_delay_pixelated':
                        spectrum = spectrum / 4 * (1 + contrast_inst[0] * np.cos(delay + phase_mask))

                    elif self.type == 'double_delay_pixelated':
                        contrast_inst_sum = contrast_inst_diff = prod(contrast_inst)
                        delay_sum, delay_diff = delay
                        spectrum = spectrum / 4 * (
                                1
                                + 0.5 * contrast_inst_diff * np.cos(delay_diff + phase_mask)
                                - 0.5 * contrast_inst_sum * np.cos(delay_sum + phase_mask)
                        )

                    elif self.type == 'triple_delay_pixelated':
                        contrast_inst_sum = contrast_inst_diff = prod(contrast_inst)
                        delay_2, delay_sum, delay_diff = delay
                        root2 = np.sqrt(2)
                        spectrum = spectrum / 4 * (
                                1
                                + root2 / 2 * contrast_inst[1] * np.cos(delay_2 + phase_mask)
                                + root2 / 4 * contrast_inst_diff * np.cos(delay_diff + phase_mask)
                                - root2 / 4 * contrast_inst_sum * np.cos(delay_sum + phase_mask)
                        )

                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            except NotImplementedError:
                failed = True
                # TODO add warning here?

        if self.type == 'mueller' or failed is True:
            # full Mueller matrix calculation
            if 'stokes' not in spectrum.dims:
                a0 = xr.zeros_like(spectrum)
                spectrum = xr.combine_nested([spectrum, a0, a0, a0], concat_dim=('stokes',))
            mueller_matrix_total = self.get_mueller_matrix(spectrum.wavelength, x, y)
            spectrum = mueller_product(mueller_matrix_total, spectrum)
            apply_polarisers = None

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
            for crystal in self.retarders:

                sp_f_x, sp_f_y = crystal.get_fringe_frequency(wavelength, self.optics[2], )
                spatial_freq_x += sp_f_x
                spatial_freq_y += sp_f_y

        elif self.type == 'multi_delay_pixelated':
            crystal = self.retarders[0]
            spatial_freq_x, spatial_freq_y = crystal.get_fringe_frequency(wavelength, self.optics[2], )
            # TODO and also the sum and difference terms?

        else:
            raise NotImplementedError

        return spatial_freq_x, spatial_freq_y

    def _check_interferometer_config(self, types, relative_orientations):

        conditions_met = []
        for idx, (typ, rel_or) in enumerate(zip(types, relative_orientations)):
            component = self.interferometer[idx]
            conditions_met.append(isinstance(component, typ))
            conditions_met.append(isclose(component.orientation - self.polarisers[0].orientation, rel_or))

        return all(conditions_met)

    def __eq__(self, inst_other):
        condition_1 = all([getattr(self, attr) == getattr(inst_other, attr) for attr in ['camera', 'optics', ]])
        condition_2 = all([c == c_other for c, c_other in zip(self.interferometer, inst_other.interferometer)])
        if condition_1 and condition_2:
            return True
        else:
            return False