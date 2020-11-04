import numpy as np
import xarray as xr
from numba import vectorize, f8
from pycis.model import calculate_dispersion


def mueller_product(mat1, mat2):
    """
    Compute the product of two Mueller matrices.

    :param xr.DataArray mat1: Mueller matrix.
    :param xr.DataArray mat2: Mueller matrix or Stokes vector.
    :return: (xr.DataArray) mat1 @ mat2, a Mueller matrix or a Stokes vector, depending on the dimensions of mat2.
    """

    if 'mueller_v' in mat2.dims and 'mueller_h' in mat2.dims:
        mat2_i = mat2.rename({'mueller_h': 'mueller_i', 'mueller_v': 'mueller_h'})
        return mat1.dot(mat2_i, dims=('mueller_h', ), ).rename({'mueller_i': 'mueller_h'})

    elif 'stokes' in mat2.dims:
        mat2_i = mat2.rename({'stokes': 'mueller_h'})
        return mat1.dot(mat2_i, dims=('mueller_h', ), ).rename({'mueller_v': 'stokes'})

    else:
        raise Exception('input not understood')


def rotation_matrix(angle):
    """
    Mueller matrix for frame rotation (anti-clockwise from x-axis).

    :param float angle: rotation angle in radians.
    :return: (xr.DataArray) Frame rotation Mueller matrix.
    """

    angle2 = 2 * angle
    rot_mat = np.array([[1, 0, 0, 0],
                        [0, np.cos(angle2), np.sin(angle2), 0],
                        [0, -np.sin(angle2), np.cos(angle2), 0],
                        [0, 0, 0, 1]])
    return xr.DataArray(rot_mat, dims=('mueller_v', 'mueller_h'), )


class Component:
    """
    Base class for interferometer component.

    """
    def __init__(self, ):
        pass

    def get_mueller_matrix(self, *args, **kwargs):
        raise NotImplementedError


class Filter(Component):
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


class OrientableComponent(Component):
    """
    Base class for interferometer component with orientation-dependent behaviour.
    """
    def __init__(self, orientation, ):
        super().__init__()
        self.orientation = orientation

    def orient(self, matrix):
        """
        Calculate component Mueller matrix at the set orientation.

        :param xr.DataArray matrix: Component Mueller matrix.
        :return: (xr.DataArray) Component Mueller matrix at the set orientation.
        """

        matrix_i = mueller_product(matrix, rotation_matrix(self.orientation))
        return mueller_product(rotation_matrix(-self.orientation), matrix_i)


class LinearPolariser(OrientableComponent):
    """
    Linear polariser.

    :param float orientation: Orientation in radians of the transmission axis relative to the x-axis.
    :param float tx_1: transmission primary component. [0, 1] - defaults to 1.
    :param float tx_2: transmission secondary (orthogonal) component. [0, 1] - defaults to 0.
    """
    def __init__(self, orientation, tx_1=1, tx_2=0, ):
        super().__init__(orientation, )

        assert 0 <= tx_1 <= 1
        assert 0 <= tx_2 <= 1
        self.tx_1 = tx_1
        self.tx_2 = tx_2

    def get_mueller_matrix(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """

        m = [[self.tx_2 ** 2 + self.tx_1 ** 2, self.tx_1 ** 2 - self.tx_2 ** 2, 0, 0],
             [self.tx_1 ** 2 - self.tx_2 ** 2, self.tx_2 ** 2 + self.tx_1 ** 2, 0, 0],
             [0, 0, 2 * self.tx_2 * self.tx_1, 0],
             [0, 0, 0, 2 * self.tx_2 * self.tx_1]]

        return self.orient(1 / 2 * xr.DataArray(m, dims=('mueller_v', 'mueller_h'), ))


class LinearRetarder(OrientableComponent):
    """
    Base class for a general linear retarder.
    """
    def __init__(self, orientation, thickness, material='a-BBO', material_source=None, contrast=1, ):
        super().__init__(orientation, )

        self.thickness = thickness
        self.material = material
        self.source = material_source
        self.contrast = contrast

    def get_mueller_matrix(self, *args, **kwargs):
        """
        General Mueller matrix for a linear retarder.
        """

        delay = self.get_delay(*args, **kwargs)

        m1s = xr.ones_like(delay)
        m0s = xr.zeros_like(delay)
        c_c = self.contrast * np.cos(delay)
        c_s = self.contrast * np.sin(delay)

        m = [[m1s,  m0s,  m0s,  m0s],
             [m0s,  m1s,  m0s,  m0s],
             [m0s,  m0s,  c_c,  c_s],
             [m0s,  m0s, -c_s,  c_c]]

        return self.orient(xr.combine_nested(m, concat_dim=('mueller_v', 'mueller_h', ), ))

    def get_delay(self, *args, **kwargs):
        """
        Interferometer delay in radians.
        """
        raise NotImplementedError

    def get_fringe_frequency(self, *args, **kwargs):
        """
        Spatial frequency of the fringe pattern at the sensor plane in units m^-1.
        """
        raise NotImplementedError


class UniaxialCrystal(LinearRetarder):
    """
    Plane-parallel uniaxial birefringent crystal.

    :param float orientation: Orientation of component fast axis in radians, relative to the x-axis.
    :param float thickness: Crystal thickness in m.
    :param float cut_angle: Angle in radians between crystal optic axis and front face.
    :param str material: Crystal material.
    :param str material_source: Source of Sellmeier coefficients describing dispersion in the crystal. If blank, the
        default material source specified in pycis.model.dispersion
    :param float contrast: An arbitrary contrast degradation factor for the retarder, independent of ray path. Value
        between 0 and 1. Simulates the effect of real crystal imperfections.
    """
    def __init__(self, orientation, thickness, cut_angle, material='a-BBO', material_source=None, contrast=1, ):
        super().__init__(orientation, thickness, material=material, material_source=material_source, contrast=contrast, )
        self.cut_angle = cut_angle

    def get_delay(self, wavelength, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        Calculate imparted delay in radians.

        :param wavelength: Wavelength in m.
        :type wavelength: float, xr.DataArray
        :param inc_angle: Ray incidence angle(s) in radians.
        :type inc_angle: float, xr.DataArray
        :param azim_angle: Ray azimuthal angle(s) in radians.
        :type azim_angle: float, xr.DataArray
        :param float n_e: Manually set extraordinary refractive index (e.g. for fitting).
        :param float n_o: Manually set ordinary refractive index (e.g. for fitting).
        :return: (float, xr.DataArray) Imparted delay in radians.
        """

        # Veiras defines optical path difference as OPL_o - OPL_e ie. +ve phase indicates a delayed extraordinary
        # ray.

        # if refractive indices have not been manually set, calculate
        if n_e is None and n_o is None:
            biref, n_e, n_o = calculate_dispersion(wavelength, self.material, source=self.source)

        args = [wavelength, inc_angle, azim_angle, n_e, n_o, self.cut_angle, self.thickness, ]
        return xr.apply_ufunc(_calc_delay_uniaxial_crystal, *args, dask='allowed', )

    def get_fringe_frequency(self, wavelength, focal_length, ):
        """
        Calculate the (approx.) spatial frequency of the fringe pattern at the sensor plane.

        :param float focal_length: Focal length of imaging lens in m.
        :param float wavelength: Wavelength in m.
        :return: (tuple) x and y components of the fringe frequency in units m^-1 and in order (f_x, f_y).
        """

        biref, n_e, n_o = calculate_dispersion(wavelength, self.material, source=self.source)

        # derived by first-order approx. of the Veiras formula.
        factor = (n_o ** 2 - n_e ** 2) * np.sin(self.cut_angle) * np.cos(self.cut_angle) / \
                 (n_e ** 2 * np.sin(self.cut_angle) ** 2 + n_o ** 2 * np.cos(self.cut_angle) ** 2)
        freq = self.thickness / (wavelength * focal_length) * factor

        freq_x = freq * np.cos(self.orientation)
        freq_y = freq * np.sin(self.orientation)

        return freq_x, freq_y


class SavartPlate(LinearRetarder):
    """
    Savart plate.

    :param float orientation: Orientation of component fast axis in radians, relative to the x-axis.
    :param float thickness: Total thickness of plate in m.
    :param str material: Crystal material.
    :param str material_source: Source of Sellmeier coefficients describing dispersion in the crystal. If blank, the
        default material source specified in pycis.model.dispersion
    :param float contrast: An arbitrary contrast degradation factor for the retarder, independent of ray path. Value
        between 0 and 1. Simulates the effect of real crystal imperfections.
    :param str mode: Determines how impartedd delay is calculated: 'francon' (approx.) or 'veiras' (exact).
    """
    def __init__(self, orientation, thickness, material='a-BBO', material_source=None, contrast=1, mode='francon', ):
        super().__init__(orientation, thickness, material=material, material_source=material_source, contrast=contrast, )
        self.mode = mode

    def get_delay(self, wavelength, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        Calculate imparted delay in radians.

        :param wavelength: Wavelength in m.
        :type wavelength: float, xr.DataArray
        :param inc_angle: Ray incidence angle(s) in radians.
        :type inc_angle: float, xr.DataArray
        :param azim_angle: Ray azimuthal angle(s) in radians.
        :type azim_angle: float, xr.DataArray
        :param float n_e: manually set extraordinary refractive index (e.g. for fitting).
        :param float n_o: manually set ordinary refractive index (e.g. for fitting).
        :return: (float, xr.DataArray) Imparted delay in radians.
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
            or2 = self.orientation - np.pi / 2

            azim_angle1 = azim_angle
            azim_angle2 = azim_angle - np.pi / 2
            t = self.thickness / 2

            crystal_1 = UniaxialCrystal(or1, t, cut_angle=-np.pi / 4, material=self.material)
            crystal_2 = UniaxialCrystal(or2, t, cut_angle=np.pi / 4, material=self.material)

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

    :param float orientation: Orientation of component fast axis in radians, relative to the x-axis.
    :param float delay: Imparted delay in radians.
    """
    def __init__(self, orientation, delay, ):
        thickness = 1.  # this value is arbitrary since delay is set
        super().__init__(orientation, thickness, )
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
    def __init__(self, orientation, ):
        ideal_delay = np.pi / 2
        super().__init__(orientation, ideal_delay, )


class HalfWaveplate(IdealWaveplate):
    """
    Ideal half-wave plate.
    """
    def __init__(self, orientation, ):
        ideal_delay = np.pi
        super().__init__(orientation, ideal_delay, )


@vectorize([f8(f8, f8, f8, f8, f8, f8, f8), ], nopython=True, fastmath=True, cache=True, )
def _calc_delay_uniaxial_crystal(wavelength, inc_angle, azim_angle, n_e, n_o, cut_angle, thickness, ):
    s_inc_angle = np.sin(inc_angle)
    s_inc_angle_2 = s_inc_angle ** 2
    s_cut_angle_2 = np.sin(cut_angle) ** 2
    c_cut_angle_2 = np.cos(cut_angle) ** 2

    term_1 = np.sqrt(n_o ** 2 - s_inc_angle_2)

    term_2 = (n_o ** 2 - n_e ** 2) * \
             (np.sin(cut_angle) * np.cos(cut_angle) * np.cos(azim_angle) * s_inc_angle) / \
             (n_e ** 2 * s_cut_angle_2 + n_o ** 2 * c_cut_angle_2)

    term_3 = - n_o * np.sqrt(
        (n_e ** 2 * (n_e ** 2 * s_cut_angle_2 + n_o ** 2 * c_cut_angle_2)) -
        ((n_e ** 2 - (n_e ** 2 - n_o ** 2) * c_cut_angle_2 * np.sin(
            azim_angle) ** 2) * s_inc_angle_2)) / \
             (n_e ** 2 * s_cut_angle_2 + n_o ** 2 * c_cut_angle_2)

    return 2 * np.pi * (thickness / wavelength) * (term_1 + term_2 + term_3)

