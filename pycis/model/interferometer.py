import numpy as np
import xarray as xr
from numba import vectorize, f8
from pycis.model import calculate_dispersion


def mueller_product(mat1, mat2):
    """
    Compute the product of two Mueller matrices.

    :param xr.DataArray mat1: Mueller matrix
    :param xr.DataArray mat2: Mueller matrix or Stokes vector
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
    :return: (xr.DataArray) rotation matrix
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

    """
    def __init__(self, tx, n):
        """
        :param xr.DataArray tx: Fractional filter transmission, whose dimension 'wavelength' has coordinates in m.
        :param float n: Refractive index (effective).
        """

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
        """
        :param orientation: float, in radians

        """
        super().__init__()
        self.orientation = orientation

    def orient(self, matrix):
        """
        :param matrix: (xr.DataArray) Mueller matrix

        :return:
        """

        matrix_i = mueller_product(matrix, rotation_matrix(self.orientation))
        return mueller_product(rotation_matrix(-self.orientation), matrix_i)


class LinearPolariser(OrientableComponent):
    """
    linear polariser

    """
    def __init__(self, orientation, tx_1=1, tx_2=0, ):
        """
        :param float orientation: Orientation in radians of thetransmission axis relative to the x-axis.
        :param tx_1: transmission primary component. [0, 1] - defaults to 1
        :param tx_2: transmission secondary (orthogonal) component. [0, 1] - defaults to 0

        """
        super().__init__(orientation, )

        assert 0 <= tx_1 <= 1
        assert 0 <= tx_2 <= 1
        self.tx_1 = tx_1
        self.tx_2 = tx_2

    def get_mueller_matrix(self, *args, **kwargs):

        m = [[self.tx_2 ** 2 + self.tx_1 ** 2, self.tx_1 ** 2 - self.tx_2 ** 2, 0, 0],
             [self.tx_1 ** 2 - self.tx_2 ** 2, self.tx_2 ** 2 + self.tx_1 ** 2, 0, 0],
             [0, 0, 2 * self.tx_2 * self.tx_1, 0],
             [0, 0, 0, 2 * self.tx_2 * self.tx_1]]

        return self.orient(1 / 2 * xr.DataArray(m, dims=('mueller_v', 'mueller_h'), ))


class LinearRetarder(OrientableComponent):
    """
    Base class for a general linear retarder

    """
    def __init__(self, orientation, thickness, material='a-BBO', material_source=None, contrast=1, ):
        """
        :param float thickness: thickness in m.
        :param str material: Crystal material
        :param str material_source: Source of Sellmeier coefficients describing dispersion in the crystal.
        If blank, the default material source specified in pycis.model.dispersion

        :param contrast: arbitrary contrast degradation factor for crystal, uniform contrast only for now.
        :type contrast: float
        """
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
        Interferometer delay in radians

        """
        raise NotImplementedError

    def get_fringe_frequency(self, *args, **kwargs):
        """
        Spatial frequency of the fringe pattern at the sensor plane in units m^-1

        """
        raise NotImplementedError


class UniaxialCrystal(LinearRetarder):
    """
    Uniaxial birefringent crystal.

    """
    def __init__(self, orientation, thickness, cut_angle, material='a-BBO', material_source=None, contrast=1, ):
        """
        :param float cut_angle: angle between optic axis and crystal front face in radians.

        """
        super().__init__(orientation, thickness, material=material, material_source=material_source, contrast=contrast, )
        self.cut_angle = cut_angle

    def get_delay(self, wavelength, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        calculate phase delay (in rad) due to uniaxial crystal.

        Veiras defines optical path difference as OPL_o - OPL_e ie. +ve phase indicates a delayed extraordinary
        ray

        :param wavelength: wavelength in m.
        :param inc_angle: ray incidence angle in radians.
        :param azim_angle: ray azimuthal angle in radians.
        :param n_e: manually set extraordinary refractive index (for fitting)
        :param n_o: manually set ordinary refractive index (for fitting)
        :return: delay in radians.
        """

        # if refractive indices have not been manually set, calculate
        if n_e is None and n_o is None:
            biref, n_e, n_o = calculate_dispersion(wavelength, self.material, source=self.source)

        args = [wavelength, inc_angle, azim_angle, n_e, n_o, self.cut_angle, self.thickness, ]
        return xr.apply_ufunc(_calc_delay_uniaxial_crystal, *args, dask='allowed', )

    def get_fringe_frequency(self, wavelength, focal_length, ):
        """
        calculate the approx. spatial frequency of the fringe pattern for a given lens focal length and light wavelength

        Calculated by first order approximation of the Veiras formula.

        :param focal_length: in m
        :param wavelength: in m
        :return: spatial frequency x, spatial frequency y in (m ** -1)
        """
        biref, n_e, n_o = calculate_dispersion(wavelength, self.material, source=self.source)

        factor = (n_o ** 2 - n_e ** 2) * np.sin(self.cut_angle) * np.cos(self.cut_angle) / \
                 (n_e ** 2 * np.sin(self.cut_angle) ** 2 + n_o ** 2 * np.cos(self.cut_angle) ** 2)
        freq = self.thickness / (wavelength * focal_length) * factor

        freq_x = freq * np.cos(self.orientation)
        freq_y = freq * np.sin(self.orientation)

        return freq_x, freq_y


class SavartPlate(LinearRetarder):
    """
    Savart plate

    """
    def __init__(self, orientation, thickness, material='a-BBO', material_source=None, contrast=1, mode='francon', ):
        """
        :param mode: source for the equation for phase delay: 'francon' (approx.) or 'veiras' (exact)
        :type mode: string

        """
        super().__init__(orientation, thickness, material=material, material_source=material_source, contrast=contrast, )
        self.mode = mode

    def get_delay(self, wavelength, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        calculate phase delay (in rad) due to Savart plate.

        Vectorised. If inc_angle and azim_angle are arrays, they must have the same dimensions.  
        source: Lei Wu, Chunmin Zhang, and Baochang Zhao. “Analysis of the lateral displacement and optical path difference
        in wide-field-of-view polarization interference imaging spectrometer”. In: Optics Communications 273.1 (2007), 
        pp. 67–73. issn: 00304018. doi: 10.1016/j.optcom.2006.12.034.

        :param wavelength: wavelength [ m ]
        :type wavelength: float or array-like

        :param inc_angle: ray incidence angle [ rad ]
        :type inc_angle: float or array-like

        :param azim_angle: ray azimuthal angle [ rad ]
        :type azim_angle: float or array-like

        :param n_e: manually set extraordinary refractive index (for fitting)
        :type n_e: float

        :param n_o: manually set ordinary refractive index (for fitting)
        :type n_o: float

        :return: phase [ rad ]

        """

        if self.mode == 'francon':

            # if refractive indices have not been manually set, calculate them using Sellmeier eqn.
            if n_e is None and n_o is None:
                biref, n_e, n_o = calculate_dispersion(wavelength, self.material, source=self.source)

            a = 1 / n_e
            b = 1 / n_o

            # precalculate trig fns
            c_azim_angle = np.cos(azim_angle)
            s_azim_angle = np.sin(azim_angle)
            s_inc_angle = np.sin(inc_angle)

            # calculation
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
        # TODO
        raise NotImplementedError


class IdealWaveplate(LinearRetarder):
    """
    Idealised waveplate imparting a single delay regardless of crystal thickness, wavelength or ray path

    """
    def __init__(self, orientation, ideal_delay, ):
        thickness = 1.  # this value is arbitrary since delay is set
        super().__init__(orientation, thickness, )
        self.ideal_delay = ideal_delay

    def get_delay(self, *args, **kwargs):
        return xr.DataArray(self.ideal_delay)

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

