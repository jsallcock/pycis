import numpy as np
import xarray as xr
from numba import vectorize, f8
from scipy.constants import c
from pycis.model import mueller_product, LinearPolariser, calculate_coherence, BirefringentComponent, \
    InterferometerComponent, Camera, calculate_rot_matrix, QuarterWaveplate, UniaxialCrystal


class Instrument(object):
    """
    coherence imaging instrument

    """

    def __init__(self, camera, optics, interferometer, interferometer_orientation=0):
        """

        :param camera:
        :type camera pycis.model.Camera

        :param optics: the focal lengths of the three lenses used in the standard CI configuration (see e.g. my thesis
        or Scott Silburn's): [f_1, f_2, f_3] where f_1 is the objective lens.
        :type optics: list of floats

        :param interferometer: a list of instances of pycis.model.InterferometerComponent. The first component in
        the list is the first component that the light passes through.
        :type interferometer: list

        """

        self.camera = camera
        self.optics = optics
        self.interferometer = interferometer
        self.crystals = self.get_crystals()
        self.polarisers = self.get_polarisers()
        self.interferometer_orientation = interferometer_orientation

        self.input_checks()

        # assign instrument 'type'
        self.instrument_type = self.check_instrument_type()

    def input_checks(self):
        assert isinstance(self.camera, Camera)
        assert isinstance(self.optics, list)
        assert all(isinstance(c, InterferometerComponent) for c in self.interferometer)

    def get_crystals(self):
        return [co for co in self.interferometer if isinstance(co, BirefringentComponent)]

    def get_polarisers(self):
        return [co for co in self.interferometer if isinstance(co, LinearPolariser)]

    def calculate_inc_angle(self, x, y):
        """
        calculate incidence angles of ray(s) through crystal

        :param x: (xr.DataArray) x position(s) on sensor plane [ m ]
        :param y: (xr.DataArray) y position(s) on sensor [ m ]
        :return: incidence angles [ rad ]

        """
        return xr.apply_ufunc(_calculate_inc_angles, x, y, self.optics[2], dask='allowed')

    def calculate_azim_angle(self, x, y, crystal):
        """
        calculate azimuthal angles of rays through crystal (varies with crystal orientation)

        :param x: (xr.DataArray) x position(s) on sensor plane [ m ]
        :param y: (xr.DataArray) y position(s) on sensor plane [ m ]
        :param crystal: (pycis.model.BirefringentComponent)
        :return: azimuthal angles [ rad ]

        """
        return xr.apply_ufunc(_calculate_azim_angles, x, y, crystal.orientation, self.interferometer_orientation,
                              dask='allowed, ')

    def calculate_matrix(self, spectrum):
        """
        calculate the total Mueller matrix for the interferometer

        :param spectrum: (xr.DataArray) see spectrum argument for instrument.capture
        :return: Mueller matrix

        """

        inc_angle = self.calculate_inc_angle(spectrum.x, spectrum.y)
        total_matrix = xr.DataArray(np.identity(4), dims=('mueller_v', 'mueller_h'), )

        for component in self.interferometer:
            azim_angle = self.calculate_azim_angle(spectrum.x, spectrum.y, component)
            component_matrix = component.calculate_matrix(spectrum.wavelength, inc_angle, azim_angle)
            total_matrix = mueller_product(component_matrix, total_matrix)

        rot_matrix = calculate_rot_matrix(self.interferometer_orientation)
        return mueller_product(rot_matrix, total_matrix)

    def capture_image(self, spectrum, ):
        """
        capture image of scene

        :param spectrum: (xr.DataArray) photon fluence spectrum with units of ph / m [hitting the pixel area during exposure
         time] and with dimensions 'wavelength', 'x', 'y' and (optionally) 'stokes'. If no stokes dim then it is assumed
        that light is unpolarised (i.e. the spec supplied is the S_0 Stokes parameter only)
        :param color: (bool) true for color display, else monochrome
        :return:

        """

        if self.instrument_type == 'single_delay_linear' and 'stokes' not in spectrum.dims:
            # analytical calculation to save time
            total_intensity = spectrum.integrate(dim='wavelength', )
            spec_freq = spectrum.rename({'wavelength': 'frequency'})
            spec_freq['frequency'] = c / spec_freq['frequency']
            spec_freq = spec_freq * c / spec_freq['frequency'] ** 2
            freq_com = (spec_freq * spec_freq['frequency']).integrate(dim='frequency') / total_intensity

            delay = self.calculate_ideal_delay(c / freq_com)

            coherence = calculate_coherence(spec_freq, delay, material=self.crystals[0].material, freq_com=freq_com)
            coherence = xr.where(total_intensity > 0, coherence, 0)
            spectrum = 1 / 4 * (total_intensity + np.real(coherence))
            apply_polarisers = False

        elif self.instrument_type == 'single_delay_polarised' and 'stokes' not in spectrum.dims:
            total_intensity = spectrum.integrate(dim='wavelength', )
            spec_freq = spectrum.rename({'wavelength': 'frequency'})
            spec_freq['frequency'] = c / spec_freq['frequency']
            spec_freq = spec_freq * c / spec_freq['frequency'] ** 2
            freq_com = (spec_freq * spec_freq['frequency']).integrate(dim='frequency') / total_intensity

            delay = self.calculate_ideal_delay(c / freq_com)

            phase_mask = self.camera.calculate_pixelated_phase_mask()

            coherence = calculate_coherence(spec_freq, delay, material=self.crystals[0].material, freq_com=freq_com)
            coherence = xr.where(total_intensity > 0, coherence, 0)
            spectrum = 1 / 4 * (total_intensity + np.real(coherence * np.exp(1j * phase_mask)))
            apply_polarisers = False

        elif self.instrument_type == 'general':
            # full Mueller matrix calculation
            if 'stokes' not in spectrum.dims:
                a0 = xr.zeros_like(spectrum)
                spectrum = xr.combine_nested([spectrum, a0, a0, a0], concat_dim=('stokes',))

            mueller_matrix_total = self.calculate_matrix(spectrum)
            spectrum = mueller_product(mueller_matrix_total, spectrum)
            apply_polarisers = None
        else:
            raise NotImplementedError

        image = self.camera.capture_image(spectrum, apply_polarisers=apply_polarisers)
        return image

    def calculate_ideal_delay(self, wavelength, ):
        """
        calculate the interferometer phase delay (in rad) at the given wavelength(s)

        assumes all crystal's phase contributions combine constructively -- method used only when instrument.type =
        'two-beam'. kwargs included for fitting purposes.

        :param wavelength: can be scalar or xr.DataArray with dimensions including 'x' and 'y'
        :return:
        """

        assert self.instrument_type in ['single_delay_linear',
                                        'single_delay_polarised',
                                        'multi_delay_polarised',
                                        ]

        # calculate the ray angles through the interferometer
        if hasattr(wavelength, 'coords'):
            if 'x' in wavelength.coords.keys() and 'y' in wavelength.coords.keys():
                inc_angle = self.calculate_inc_angle(wavelength.x, wavelength.y)
                azim_angle = self.calculate_azim_angle(wavelength.x, wavelength.y, self.crystals[0])
        else:
            x, y, =  self.camera.calculate_pixel_position()
            inc_angle = self.calculate_inc_angle(x, y, )
            azim_angle = self.calculate_azim_angle(x, y, self.crystals[0])

        if self.instrument_type in ['single_delay_linear', 'single_delay_polarised']:
            # calculate phase delay contribution due to each crystal
            delay = 0
            for crystal in self.crystals:
                delay += crystal.calculate_delay(wavelength, inc_angle, azim_angle, )

        elif self.instrument_type == 'multi_delay_polarised':
            delay_1 = self.crystals[0].calculate_delay(wavelength, inc_angle, azim_angle, )
            delay_2 = self.crystals[1].calculate_delay(wavelength, inc_angle, azim_angle, )
            delay_sum = delay_1 + delay_2
            delay_diff = abs(delay_1 - delay_2)

            return delay_1, delay_2, delay_sum, delay_diff
        else:
            raise NotImplementedError

        return delay

    def calculate_ideal_contrast(self):

        contrast = 1
        for crystal in self.crystals:
            contrast *= crystal.contrast

        return contrast

    def check_instrument_type(self):
        """
        For certain instrument layouts there are analytical shortcuts for calculating the interferogram, skipping the
        full Mueller matrix treatment.

        in the case of a perfectly aligned coherence imaging diagnostic in a simple 'two-beam' configuration, skip the
        Mueller matrix calculation to the final result.

        :return: itype (str)
        """

        itype = None

        if self.camera.polarised:

            # single-delay polarised
            if len(self.interferometer) == 3:
                types = [LinearPolariser, UniaxialCrystal, QuarterWaveplate, ]
                orientations = [0, np.pi / 4, np.pi / 2, ]

                conditions = []
                for idx, (typ, orientation) in enumerate(zip(types, orientations)):
                    interferometer_component = self.interferometer[idx]
                    conditions.append(isinstance(interferometer_component, typ))
                    conditions.append(interferometer_component.orientation == orientation)

                if all(conditions):
                    itype = 'single_delay_polarised'

            # multi-delay polarised
            # TODO add option for a Savart system instead of a displacer system
            elif len(self.interferometer) == 5:
                types = [LinearPolariser, UniaxialCrystal, LinearPolariser, UniaxialCrystal, QuarterWaveplate, ]
                orientations = [0, np.pi / 4, 0, np.pi / 4, np.pi / 2, ]

                conditions = []
                for idx, (typ, orientation) in enumerate(zip(types, orientations)):
                    interferometer_component = self.interferometer[idx]
                    conditions.append(isinstance(interferometer_component, typ))
                    conditions.append(interferometer_component.orientation == orientation)

                if all(conditions):
                    itype = 'multi_delay_polarised'

        # are there two polarisers, at the front and back of the interferometer?
        elif len(self.polarisers) == 2 and (isinstance(self.interferometer[0], LinearPolariser) and
                                            isinstance(self.interferometer[-1], LinearPolariser)):

            pol_1_orientation = self.interferometer[0].orientation
            pol_2_orientation = self.interferometer[-1].orientation

            conditions = [pol_1_orientation == pol_2_orientation, ] + \
                         [abs(crys.orientation - pol_1_orientation) == np.pi / 4 for crys in self.crystals]
            if all(conditions):
                itype = 'single_delay_linear'

        if itype is None:
            itype = 'general'

        return itype

    def calculate_ideal_phase_offset(self, wl, n_e=None, n_o=None):
        """
        :param wl:
        :param n_e:
        :param n_o:
        :return: phase_offset [ rad ]
        """

        phase_offset = 0
        for crystal in self.crystals:
            phase_offset += crystal.calculate_delay(wl, 0., 0., n_e=n_e, n_o=n_o)

        return -phase_offset


@vectorize([f8(f8, f8, f8, ), ], nopython=True, fastmath=True, cache=True, )
def _calculate_inc_angles(x, y, f_3):
    return np.arctan2((x ** 2 + y ** 2) ** 0.5, f_3, )


@vectorize([f8(f8, f8, f8, f8), ], nopython=True, fastmath=True, cache=True, )
def _calculate_azim_angles(x, y, crystal_orientation, interferometer_orientation, ):
    orientation = crystal_orientation + interferometer_orientation
    return np.arctan2(y, x) + np.pi - orientation
