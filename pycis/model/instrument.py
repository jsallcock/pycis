from math import isclose
import numpy as np
import xarray as xr
from numba import vectorize, f8
from scipy.constants import c
from pycis.model import mueller_product, LinearPolariser, calculate_coherence, BirefringentComponent, \
    InterferometerComponent, Camera, QuarterWaveplate, UniaxialCrystal


class Instrument(object):
    """
    coherence imaging instrument

    """
    def __init__(self, camera, optics, interferometer, force_mueller=False):
        """

        :param camera: instance of pycis.model.Camera

        :param optics: list of floats, the focal lengths (in m) of the three lenses used in the standard CI
        configuration (see e.g. my thesis or Scott Silburn's): [f_1, f_2, f_3] where f_1 is the objective lens.

        :param interferometer:  list of instances of pycis.model.InterferometerComponent, where the first list entry is
         the first component that the light passes through.

        :param force_mueller: bool, forces the full Mueller matrix calculation of the interferogram, even when
        an analytical shortcut is available.
        """

        self.camera = camera
        self.optics = optics
        self.interferometer = interferometer
        self.force_mueller = force_mueller

        self.input_checks()
        self.crystals = [co for co in self.interferometer if isinstance(co, BirefringentComponent)]
        self.polarisers = [co for co in self.interferometer if isinstance(co, LinearPolariser)]
        self.instrument_type = self.get_instrument_type()

    def input_checks(self):
        assert isinstance(self.camera, Camera)
        assert isinstance(self.optics, list)
        assert all(isinstance(co, InterferometerComponent) for co in self.interferometer)

    def get_instrument_type(self):
        """
        For certain instrument layouts there are analytical shortcuts for calculating the interferogram, skipping the
        full Mueller matrix treatment.

        current instrument types:
        - 'mueller': Full Mueller calculation
        - 'single_delay_linear'
        - 'single_delay_polarised'
        - 'multi_delay_polarised'

        :return: itype (str)
        """

        itype = None

        if self.camera.polarised:

            # single-delay polarised
            if len(self.interferometer) == 3:

                # check for correct component types and relative orientations
                types = [LinearPolariser, UniaxialCrystal, QuarterWaveplate, ]
                relative_orientations = [0, np.pi / 4, np.pi / 2, ]

                conditions_met = []
                for idx, (typ, rel_or) in enumerate(zip(types, relative_orientations)):
                    component = self.interferometer[idx]
                    conditions_met.append(isinstance(component, typ))
                    conditions_met.append(isclose(component.orientation - self.polarisers[0].orientation, rel_or))

                if all(conditions_met):
                    itype = 'single_delay_polarised'

            # multi-delay polarised
            elif len(self.interferometer) == 5:

                # check for correct component types and relative orientations
                types = [LinearPolariser, UniaxialCrystal, LinearPolariser, UniaxialCrystal, QuarterWaveplate, ]
                relative_orientations = [0, np.pi / 4, 0, np.pi / 4, np.pi / 2, ]

                conditions_met = []
                for idx, (typ, rel_or) in enumerate(zip(types, relative_orientations)):
                    component = self.interferometer[idx]
                    conditions_met.append(isinstance(component, typ))
                    conditions_met.append(isclose(component.orientation - self.polarisers[0].orientation, rel_or))

                if all(conditions_met):
                    itype = 'multi_delay_polarised'

        # are there two polarisers, at the front and back of the interferometer?
        elif len(self.polarisers) == 2 and (isinstance(self.interferometer[0], LinearPolariser) and
                                            isinstance(self.interferometer[-1], LinearPolariser)):

            pol_1_orientation = self.interferometer[0].orientation
            pol_2_orientation = self.interferometer[-1].orientation

            conditions_met = [pol_1_orientation == pol_2_orientation, ] + \
                             [isclose(crys.orientation - pol_1_orientation, np.pi / 4) for crys in self.crystals]
            if all(conditions_met):
                itype = 'single_delay_linear'

        if itype is None or self.force_mueller:
            itype = 'mueller'

        return itype

    def calculate_inc_angle(self, x, y):
        """
        calculate incidence angles of ray(s) through crystal

        :param x: (xr.DataArray) x position(s) on sensor plane [ m ]
        :param y: (xr.DataArray) y position(s) on sensor [ m ]
        :return: incidence angles [ rad ]

        """
        return xr.apply_ufunc(_calculate_inc_angle, x, y, self.optics[2], dask='allowed')

    def calculate_azim_angle(self, x, y, crystal):
        """
        calculate azimuthal angles of rays through crystal (varies with crystal orientation)

        :param x: (xr.DataArray) x position(s) on sensor plane [ m ]
        :param y: (xr.DataArray) y position(s) on sensor plane [ m ]
        :param crystal: (pycis.model.BirefringentComponent)
        :return: azimuthal angles [ rad ]

        """
        return xr.apply_ufunc(_calculate_azim_angle, x, y, crystal.orientation, dask='allowed, ')

    def calculate_mueller_matrix(self, spectrum):
        """
        calculate the total Mueller matrix for the interferometer

        :param spectrum: (xr.DataArray) see spectrum argument for instrument.capture
        :return: Mueller matrix

        """

        inc_angle = self.calculate_inc_angle(spectrum.x, spectrum.y)
        total_matrix = xr.DataArray(np.identity(4), dims=('mueller_v', 'mueller_h'), )

        for component in self.interferometer:
            azim_angle = self.calculate_azim_angle(spectrum.x, spectrum.y, component)
            component_matrix = component.calculate_mueller_matrix(spectrum.wavelength, inc_angle, azim_angle)
            total_matrix = mueller_product(component_matrix, total_matrix)

        return total_matrix

    def capture_image(self, spectrum, ):
        """
        capture image of scene

        :param spectrum: (xr.DataArray) photon fluence spectrum with units of ph / m [hitting the pixel area during
        exposure time] and with dimensions 'wavelength', 'x', 'y' and (optionally) 'stokes'. If no stokes dim then it is assumed
        that light is unpolarised (i.e. the spec supplied is the S_0 Stokes parameter only)

        :return:

        """

        if self.instrument_type == 'mueller':
            # do the full Mueller matrix calculation

            if 'stokes' not in spectrum.dims:
                a0 = xr.zeros_like(spectrum)
                spectrum = xr.combine_nested([spectrum, a0, a0, a0], concat_dim=('stokes',))

            mueller_matrix_total = self.calculate_mueller_matrix(spectrum)
            image = mueller_product(mueller_matrix_total, spectrum)
            apply_polarisers = None

        else:
            total_intensity = spectrum.integrate(dim='wavelength', )

            spec_freq = spectrum.rename({'wavelength': 'frequency'})
            spec_freq['frequency'] = c / spec_freq['frequency']
            spec_freq = spec_freq * c / spec_freq['frequency'] ** 2
            spec_freq = spec_freq.sortby(spec_freq.frequency)  # ensure that integration limits are from -ve to +ve frequency
            freq_com = (spec_freq * spec_freq['frequency']).integrate(dim='frequency') / total_intensity

            delay = self.calculate_delay(c / freq_com)
            apply_polarisers = False

            if self.instrument_type == 'single_delay_linear' and 'stokes' not in spectrum.dims:

                coherence = calculate_coherence(spec_freq, delay, material=self.crystals[0].material, freq_com=freq_com)
                coherence = xr.where(total_intensity > 0, coherence, 0)
                contrast = np.array([crystal.contrast for crystal in self.crystals]).prod()
                image = 1 / 4 * (total_intensity + contrast * np.real(coherence))

            elif self.instrument_type == 'single_delay_polarised' and 'stokes' not in spectrum.dims:

                phase_mask = self.camera.calculate_pixelated_phase_mask()
                coherence = calculate_coherence(spec_freq, delay, material=self.crystals[0].material, freq_com=freq_com)
                coherence = xr.where(total_intensity > 0, coherence, 0)
                image = 1 / 4 * (total_intensity + self.crystals[0].contrast * np.real(coherence * np.exp(1j * phase_mask)))

            elif self.instrument_type == 'multi_delay_polarised' and 'stokes' not in spectrum.dims:

                phase_mask = self.camera.calculate_pixelated_phase_mask()

                delay_1, delay_2, delay_sum, delay_diff = delay

                # TODO not sure how to handle this if the two crystal materials are different
                coherence_1 = calculate_coherence(spec_freq, delay_1, material=self.crystals[0].material, freq_com=freq_com)
                coherence_2 = calculate_coherence(spec_freq, delay_2, material=self.crystals[1].material, freq_com=freq_com)
                coherence_sum = calculate_coherence(spec_freq, delay_sum, material=self.crystals[0].material, freq_com=freq_com)
                coherence_diff = calculate_coherence(spec_freq, delay_diff, material=self.crystals[0].material, freq_com=freq_com)

                contrast_1 = self.crystals[0].contrast
                contrast_2 = self.crystals[0].contrast
                contrast_sum = contrast_diff = contrast_1 * contrast_2

                image = 1 / 8 * (total_intensity +
                                 contrast_1 * np.real(coherence_1) +
                                 contrast_2 * np.real(coherence_2 * np.exp(1j * phase_mask)) +
                                 1 / 2 * contrast_sum * np.real(coherence_sum * np.exp(1j * phase_mask)) +
                                 1 / 2 * contrast_diff * np.real(coherence_diff * np.exp(1j * phase_mask))
                                 )
            else:
                raise NotImplementedError

        # image = xr.where(image <= 0, 0, image, )
        image = self.camera.capture_image(image, apply_polarisers=apply_polarisers)
        return image

    def calculate_delay(self, wavelength, ):
        """
        calculate the interferometer delay(s) at the given wavelength(s)

        At the moment this method only works for instrument types other than 'mueller'. I'm not sure it would be
        possible to write a general function.

        :param wavelength: in units (m), can be a float or an xr.DataArray with dimensions including 'x' and 'y'
        :return: delay in units (rad)
        """

        assert self.instrument_type != 'mueller'

        # calculate the ray angles through the interferometer
        if hasattr(wavelength, 'coords'):
            if 'x' in wavelength.coords.keys() and 'y' in wavelength.coords.keys():
                inc_angle = self.calculate_inc_angle(wavelength.x, wavelength.y)
                azim_angle = self.calculate_azim_angle(wavelength.x, wavelength.y, self.crystals[0])
        else:
            inc_angle = self.calculate_inc_angle(self.camera.x, self.camera.y, )
            azim_angle = self.calculate_azim_angle(self.camera.x, self.camera.y, self.crystals[0])

        if self.instrument_type == 'single_delay_linear':
            # add delay contribution due to each crystal
            delay = 0
            for crystal in self.crystals:
                delay += crystal.calculate_delay(wavelength, inc_angle, azim_angle, )

        elif self.instrument_type == 'single_delay_polarised':
            # generalise to arbitrary interferometer orientations
            orientation_delay = -2 * self.polarisers[0].orientation
            delay = self.crystals[0].calculate_delay(wavelength, inc_angle, azim_angle, ) + orientation_delay

        elif self.instrument_type == 'multi_delay_polarised':
            # generalise to arbitrary interferometer orientations
            orientation_delay = -2 * self.polarisers[0].orientation
            delay_1 = self.crystals[0].calculate_delay(wavelength, inc_angle, azim_angle, )
            delay_2 = self.crystals[1].calculate_delay(wavelength, inc_angle, azim_angle, ) + orientation_delay
            delay_sum = delay_1 + delay_2
            delay_diff = abs(delay_1 - delay_2)

            delay = delay_1, delay_2, delay_sum, delay_diff
        else:
            raise NotImplementedError

        return delay

    def calculate_fringe_frequency(self, wavelength):
        """
        calculates the (rough) interference fringe period at the sensor plane and at the given wavelength

        only makes sense for instrument types with a phase shear (e.g. 'single_delay_linear' and
        'multi_delay_polarised')

        :return:
        """
        assert self.instrument_type != 'mueller'

        if self.instrument_type == 'single_delay_linear':
            # add contribution due to each crystal
            spatial_freq_x, spatial_freq_y = 0
            for crystal in self.crystals:

                sp_f_x, sp_f_y = crystal.calculate_fringe_frequency(wavelength, self.optics[2], )
                spatial_freq_x += sp_f_x
                spatial_freq_y += sp_f_y

        elif self.instrument_type == 'multi_delay_polarised':
            crystal = self.crystals[0]
            spatial_freq_x, spatial_freq_y = crystal.calculate_fringe_frequency(wavelength, self.optics[2], )

            # TODO and also the sum and difference terms!

        else:
            raise NotImplementedError


        return spatial_freq_x, spatial_freq_y





@vectorize([f8(f8, f8, f8, ), ], nopython=True, fastmath=True, cache=True, )
def _calculate_inc_angle(x, y, f_3):
    return np.arctan2((x ** 2 + y ** 2) ** 0.5, f_3, )


@vectorize([f8(f8, f8, f8, ), ], nopython=True, fastmath=True, cache=True, )
def _calculate_azim_angle(x, y, crystal_orientation, ):
    return np.arctan2(y, x) + np.pi - crystal_orientation
