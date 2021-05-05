import numpy as np
import xarray as xr
from numba import vectorize, f8
from pycis.model import get_refractive_indices
from math import radians


def mueller_product(mat1, mat2):
    """
    Compute the product of a Mueller matrix with a Mueller matrix / Stokes vector

    :param xarray.DataArray mat1: Mueller matrix.
    :param xarray.DataArray mat2: Mueller matrix or Stokes vector.
    :return: (xarray.DataArray) mat1 @ mat2, a Mueller matrix or a Stokes vector, depending on the dimensions of mat2.
    """

    if 'mueller_v' in mat2.dims and 'mueller_h' in mat2.dims:
        mat2_i = mat2.rename({'mueller_h': 'mueller_i', 'mueller_v': 'mueller_h'})
        return mat1.dot(mat2_i, dims=('mueller_h', ), ).rename({'mueller_i': 'mueller_h'})

    elif 'stokes' in mat2.dims:
        mat2_i = mat2.rename({'stokes': 'mueller_h'})
        return mat1.dot(mat2_i, dims=('mueller_h', ), ).rename({'mueller_v': 'stokes'})

    else:
        raise ValueError('pycis: arguments not understood')


def rotation_matrix(angle):
    """
    Mueller matrix for frame rotation (anti-clockwise from x-axis)

    :param float angle: rotation angle in degrees.
    :return: (xr.DataArray) Frame rotation Mueller matrix.
    """

    angle2 = 2 * radians(angle)
    rot_mat = np.array([[1, 0, 0, 0],
                        [0, np.cos(angle2), np.sin(angle2), 0],
                        [0, -np.sin(angle2), np.cos(angle2), 0],
                        [0, 0, 0, 1]])
    return xr.DataArray(rot_mat, dims=('mueller_v', 'mueller_h'), )


class Component:
    """
    Base class for interferometer component

    """

    def __eq__(self, other_component):
        if type(self) == type(other_component) \
                and list(vars(self).values()) == list(vars(other_component).values()):
            return True
        else:
            return False


class OrientableComponent(Component):
    """
    Base class for component with orientation-dependent behaviour

    """
    def __init__(self, orientation=0, **kwargs):
        super().__init__(**kwargs)
        self.orientation = orientation

    def orient(self, matrix):
        """
        Calculate component Mueller matrix at the set orientation.

        :param xr.DataArray matrix: Component Mueller matrix.
        :return: (xr.DataArray) Component Mueller matrix at the set orientation.
        """

        matrix_i = mueller_product(matrix, rotation_matrix(self.orientation))
        return mueller_product(rotation_matrix(-self.orientation), matrix_i)


class TiltableComponent(Component):
    """
    Base class for component with tilt-dependent behaviour.

    """
    def __init__(self, tilt_x=0, tilt_y=0, **kwargs):
        super().__init__(**kwargs)
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y


class Filter(TiltableComponent):
    """
    Optical filter with no polarisation-dependent behaviour.

    :param xr.DataArray tx: Fractional filter transmission, whose dimension 'wavelength' has coordinates in m.
    :param float n: Refractive index (effective).
    """
    def __init__(self, tx, n):

        super().__init__()
        assert all(tx >= 0) and all(tx <= 1)
        self.tx = tx
        self.wl_centre = float((tx * tx.wavelength).integrate(dim='wavelength') / tx.integrate(dim='wavelength'))
        self.n = n

    def get_mueller_matrix(self, wavelength, *args, **kwargs):
        # TODO incorporate filter tilt wavelength shift
        # wl_shift = self.wl_centre * (np.sqrt(1 - (np.sin(inc_angle) / self.n) ** 2) - 1)

        tx = self.tx.interp(wavelength=wavelength)
        return xr.DataArray(np.identity(4), dims=('mueller_v', 'mueller_h',)) * tx


class LinearPolariser(OrientableComponent):
    """
    Linear polariser

    :param float orientation: \
        Orientation in degrees of the transmission axis relative to the x-axis.

    :param float tx_1: \
        Transmission, primary component. [0, 1] - default is 1.

    :param float tx_2: \
        Transmission, secondary (orthogonal) component. [0, 1] - default is 0.

    """
    def __init__(self, tx_1=1, tx_2=0, **kwargs):
        super().__init__(**kwargs)

        assert 0 <= tx_1 <= 1
        assert 0 <= tx_2 <= 1
        self.tx_1 = tx_1
        self.tx_2 = tx_2

    def get_mueller_matrix(self, *args, **kwargs):
        """
        Mueller matrix for a linear polariser

        """

        m = [[self.tx_2 ** 2 + self.tx_1 ** 2, self.tx_1 ** 2 - self.tx_2 ** 2, 0, 0],
             [self.tx_1 ** 2 - self.tx_2 ** 2, self.tx_2 ** 2 + self.tx_1 ** 2, 0, 0],
             [0, 0, 2 * self.tx_2 * self.tx_1, 0],
             [0, 0, 0, 2 * self.tx_2 * self.tx_1]]

        return self.orient(1 / 2 * xr.DataArray(m, dims=('mueller_v', 'mueller_h'), ))


class LinearRetarder(OrientableComponent, TiltableComponent):
    """
    Base class for a general linear retarder
    """
    def __init__(self, contrast=1, **kwargs):
        super().__init__(**kwargs)
        self.contrast = contrast

    def get_mueller_matrix(self, *args, **kwargs):
        """
        Mueller matrix for a linear retarder
        """

        delay = self.get_delay(*args, **kwargs)

        m1 = xr.ones_like(delay)
        m0 = xr.zeros_like(delay)
        cc = self.contrast * np.cos(delay)
        cs = self.contrast * np.sin(delay)

        m = [[m1,  m0,  m0,  m0],
             [m0,  m1,  m0,  m0],
             [m0,  m0,  cc,  cs],
             [m0,  m0, -cs,  cc]]

        return self.orient(xr.combine_nested(m, concat_dim=('mueller_v', 'mueller_h', ), ))

    def get_delay(self, *args, **kwargs):
        raise NotImplementedError

    def get_fringe_frequency(self, *args, **kwargs):
        raise NotImplementedError


class UniaxialCrystal(LinearRetarder):
    """
    Plane-parallel uniaxial birefringent crystal plate

    :param float thickness: \
        Crystal thickness in m.

    :param float cut_angle: \
        Angle in degrees between crystal optic axis and front face.

    :param float orientation: \
        Orientation of component fast axis in degrees, from positive x-axis towards positive y-axis.

    :param str material: \
        Set crystal material.

    :param str sellmeier_coefs_source: \
        Specify which source to use for the Sellmeier coefficients that describe the dispersion. If not specified,
        defaults for each material are set by sellmeier_coefs_source_defaults in pycis.model.dispersion.

    :param dict sellmeier_coefs: \
        Manually set the coefficients that describe the material dispersion
        via the Sellmeier equation. Dictionary must have keys 'Ae', 'Be', 'Ce', 'De', 'Ao', 'Bo', 'Co', 'Do'.

    :param float contrast: An arbitrary contrast degradation factor for the retarder, independent of ray path. Value
        between 0 and 1. Simulates real crystal imperfections.

    """
    def __init__(self, thickness, cut_angle, material='a-BBO', sellmeier_coefs_source=None, sellmeier_coefs=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.thickness = thickness
        self.cut_angle = cut_angle
        self.material = material
        self.sellmeier_coefs_source = sellmeier_coefs_source
        self.sellmeier_coefs = sellmeier_coefs

        if all([attr is not None for attr in [self.sellmeier_coefs_source, self.sellmeier_coefs]]):
            raise ValueError('pycis: arguments not understood')

    def get_delay(self, wavelength, inc_angle, azim_angle):
        """
        Calculate path delay (in radians) imparted by the retarder

        :param wavelength: Wavelength in m.
        :type wavelength: float, xarray.DataArray

        :param inc_angle: Ray incidence angle(s) in radians.
        :type inc_angle: float, xarray.DataArray

        :param azim_angle: Ray azimuthal angle(s) in radians.
        :type azim_angle: float, xarray.DataArray

        :return: (float, xarray.DataArray) Imparted delay in radians.

        """

        kwargs = {
            'sellmeier_coefs_source': self.sellmeier_coefs_source,
            'sellmeier_coefs': self.sellmeier_coefs,
        }
        ne, no = get_refractive_indices(wavelength, self.material, **kwargs)
        args = [wavelength, inc_angle, azim_angle, ne, no, radians(self.cut_angle), self.thickness, ]
        return xr.apply_ufunc(_calc_delay_uniaxial_crystal, *args, dask='allowed', )

    def get_fringe_frequency(self, wavelength, focal_length):
        """
        Calculate the (approx.) spatial frequency of the fringe pattern at the sensor plane.

        :param float focal_length: Focal length of imaging lens in m.
        :param float wavelength: Wavelength in m.
        :return: (tuple) x and y components of the fringe frequency in units m^-1 and in order (f_x, f_y).
        """

        kwargs = {
            'sellmeier_coefs_source': self.sellmeier_coefs_source,
            'sellmeier_coefs': self.sellmeier_coefs,
        }
        ne, no = get_refractive_indices(wavelength, self.material, **kwargs)

        # from first-order approx. of the Veiras formula.
        factor = (no ** 2 - ne ** 2) * np.sin(radians(self.cut_angle)) * np.cos(radians(self.cut_angle)) / \
                 (ne ** 2 * np.sin(radians(self.cut_angle)) ** 2 + no ** 2 * np.cos(radians(self.cut_angle)) ** 2)
        freq = self.thickness / (wavelength * focal_length) * factor

        freq_x = freq * np.cos(self.orientation)
        freq_y = freq * np.sin(self.orientation)

        return freq_x, freq_y


class SavartPlate(LinearRetarder):
    """
    Savart plate

    :param float thickness: \
        Total thickness of plate in m.

    :param float orientation: \
        Orientation of component fast axis in degrees, relative to the x-axis.

    :param str material: Crystal material.

    :param str material_source: Source of Sellmeier coefficients describing dispersion in the crystal. If blank, the
        default material source specified in pycis.model.dispersion

    :param float contrast: An arbitrary contrast degradation factor for the retarder, independent of ray path. Value
        between 0 and 1. Simulates the effect of real crystal imperfections.
    :param str mode: Determines how impartedd delay is calculated: 'francon' (approx.) or 'veiras' (exact).
    """
    def __init__(self, thickness, material='a-BBO', material_source=None, mode='francon', **kwargs):
        super().__init__(**kwargs)

        self.thickness = thickness
        self.material = material
        self.material_source = material_source,
        self.mode = mode

    def get_delay(self, wavelength, inc_angle, azim_angle):
        """
        Calculate path delay (in radians) imparted by the retarder

        :param wavelength: Wavelength in m.
        :type wavelength: float, xarray.DataArray
        :param inc_angle: Ray incidence angle(s) in radians.
        :type inc_angle: float, xarray.DataArray
        :param azim_angle: Ray azimuthal angle(s) in radians.
        :type azim_angle: float, xarray.DataArray
        :return: (float, xarray.DataArray) Imparted delay in radians.
        """

        if self.mode == 'francon':

            if n_e is None and n_o is None:
                biref, n_e, n_o = calculate_dispersion(wavelength, self.material, source=self.source)

            # Delay eqn. from Francon and Mallick's 'Polarization Interferometers' textbook.
            a = 1 / n_e
            b = 1 / n_o

            c_azim_angle = np.cos(azim_angle)
            s_azim_angle = np.sin(azim_angle)
            s_inc_angle = np.sin(inc_angle)

            term_1 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2)) * (c_azim_angle + s_azim_angle) * s_inc_angle

            term_2 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2) ** (3 / 2)) * ((a ** 2) / np.sqrt(2)) * \
                     (c_azim_angle ** 2 - s_azim_angle ** 2) * s_inc_angle ** 2

            # minus sign here makes the OPD calculation consistent with Veiras' definition
            delay = 2 * np.pi * - (self.thickness / (2 * wavelength)) * (term_1 + term_2)

        elif self.mode == 'veiras':
            # explicitly model plate as the combination of two uniaxial crystals

            or1 = self.orientation
            or2 = self.orientation - 90

            azim_angle1 = azim_angle
            azim_angle2 = azim_angle - np.pi / 2
            t = self.thickness / 2

            crystal_1 = UniaxialCrystal(or1, t, cut_angle=-45, material=self.material)
            crystal_2 = UniaxialCrystal(or2, t, cut_angle=45, material=self.material)

            delay = crystal_1.get_delay(wavelength, inc_angle, azim_angle1, n_e=n_e, n_o=n_o) - \
                    crystal_2.get_delay(wavelength, inc_angle, azim_angle2, n_e=n_e, n_o=n_o)

        else:
            raise Exception('invalid SavartPlate.mode')

        return delay

    def get_fringe_frequency(self, *args, **kwargs):
        # TODO!
        raise NotImplementedError


class IdealWaveplate(LinearRetarder):
    """
    Ideal waveplate imparting a given delay to all rays.

    :param float orientation: Orientation of component fast axis in degrees, relative to the x-axis.
    :param float delay: Imparted delay in radians.
    """
    def __init__(self, delay, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay

    def get_delay(self, *args, **kwargs):
        return xr.DataArray(self.delay)

    def get_fringe_frequency(self, *args, **kwargs):
        # no phase change across sensor plane
        return 0, 0


class QuarterWaveplate(IdealWaveplate):
    """
    Ideal quarter-wave plate.
    """
    def __init__(self, **kwargs):
        delay = np.pi / 2
        if 'delay' in kwargs:
            kwargs.pop('delay')  # nasty kludge to make instrument.write_config() work
        super().__init__(delay, **kwargs)


class HalfWaveplate(IdealWaveplate):
    """
    Ideal half-wave plate.
    """
    def __init__(self, orientation, ):
        delay = np.pi
        if 'delay' in kwargs:
            kwargs.pop('delay')  # nasty kludge to make instrument.write_config() work
        super().__init__(delay, **kwargs)


@vectorize([f8(f8, f8, f8, f8, f8, f8, f8), ], nopython=True, fastmath=True, cache=True, )
def _calc_delay_uniaxial_crystal(wavelength, inc_angle, azim_angle, ne, no, cut_angle, thickness, ):
    s_inc_angle = np.sin(inc_angle)
    s_inc_angle_2 = s_inc_angle ** 2
    s_cut_angle_2 = np.sin(cut_angle) ** 2
    c_cut_angle_2 = np.cos(cut_angle) ** 2

    term_1 = np.sqrt(no ** 2 - s_inc_angle_2)
    term_2 = (no ** 2 - ne ** 2) * \
             (np.sin(cut_angle) * np.cos(cut_angle) * np.cos(azim_angle) * s_inc_angle) / \
             (ne ** 2 * s_cut_angle_2 + no ** 2 * c_cut_angle_2)
    term_3 = - no * np.sqrt(
        (ne ** 2 * (ne ** 2 * s_cut_angle_2 + no ** 2 * c_cut_angle_2)) -
        ((ne ** 2 - (ne ** 2 - no ** 2) * c_cut_angle_2 * np.sin(
            azim_angle) ** 2) * s_inc_angle_2)) / \
             (ne ** 2 * s_cut_angle_2 + no ** 2 * c_cut_angle_2)

    return 2 * np.pi * (thickness / wavelength) * (term_1 + term_2 + term_3)

