import os
import yaml
import numpy as np
from scipy.constants import h, c, e, physical_constants
import matplotlib.pyplot as plt
from raysect.core.math.function.float import Arg1D, Constant1D


dir_path = os.path.dirname(os.path.realpath(__file__))


def zeeman_cherab():
    """
    Zeeman-split line component wavelengths, relative intensities and polarisation states.

    Weak-field anomolous Zeeman effect. Assumes that ions are well described by L-S coupling.

    Based on 'specline.m' script by Scott Silburn

    param: float bfield: the strength of the magnetic field

    param: float view: the angle in degrees between the magnetic field vector and the view vector.

    :return:
    - wavelengths: list of wavelengths
    - relative_intensities: list of relative intensities.

    TODO: this script is unfinished, need to account for view_angle and polarisation effects
    """

    mu_b = physical_constants['Bohr magneton in eV/T'][0]

    with open(os.path.join(dir_path, 'ciii.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    no_transitions = len(config)

    pi_components = []
    sigma_plus_components = []
    sigma_minus_components = []

    # loop over each transition
    for ii in range(no_transitions):
        # normalisation constant for each transition
        if ii == 0:
            norm_const = 2
        elif ii == 1:
            norm_const = 4
        elif ii == 2:
            norm_const = 20

        j_u = config[ii]['j_u']
        j_l = config[ii]['j_l']
        g_u = config[ii]['g_u']
        g_l = config[ii]['g_l']
        energy_u = config[ii]['energy_u']
        energy_l = config[ii]['energy_l']
        rel_int_fine_structure = config[ii]['rel_int']

        # Loop over total angular momentum projection mJ of upper and lower levels:
        for mj_u in np.linspace(-1 * j_u, j_u, 2 * j_u + 1):  # 2 * j_u + 1 is spin multiplicity of energy level
            for mj_l in np.linspace(-1 * j_l, j_l, 2 * j_l + 1):
                delta_j = j_u - j_l
                delta_mj = mj_u - mj_l

                # selection rule satisfied?
                if abs(delta_mj) > 1:
                    continue

                # zeeman perturbed energy states:
                energy_u_ii = energy_u + (mj_u * mu_b * Arg1D() * g_u)  # [eV]
                energy_l_ii = energy_l + (mj_l * mu_b * Arg1D() * g_l)  # [eV]
                wavelength = 1e9 * (h * c / ((energy_u_ii - energy_l_ii) * e))  # [nm]

                # print('f', rel_int_fine_structure)

                # calculate relative line intensities, expressions taken from Table 2.1, p23, S.Silburn's thesis
                if delta_j == 1:
                    if delta_mj == 1:
                        rel_int = 0.25 * (j_u + mj_u) * (j_u - 1 + mj_u)
                        wl_int_tuple = (wavelength, Constant1D(rel_int_fine_structure * (rel_int / norm_const)))
                        sigma_plus_components.append(wl_int_tuple)

                    elif delta_mj == -1:
                        rel_int = 0.25 * (j_u - mj_u) * (j_u - 1 - mj_u)
                        wl_int_tuple = (wavelength, Constant1D(rel_int_fine_structure * (rel_int / norm_const)))
                        sigma_minus_components.append(wl_int_tuple)

                    elif delta_mj == 0:
                        rel_int = j_u ** 2 - mj_u ** 2
                        wl_int_tuple = (wavelength, Constant1D(rel_int_fine_structure * (rel_int / norm_const)))
                        pi_components.append(wl_int_tuple)

                elif delta_j == -1:
                    if delta_mj == 1:
                        rel_int = 0.25 * (j_u + 1 - mj_u) * (j_u + 2 - mj_u)
                        wl_int_tuple = (wavelength, Constant1D(rel_int_fine_structure * (rel_int / norm_const)))
                        sigma_plus_components.append(wl_int_tuple)

                    elif delta_mj == -1:
                        rel_int = 0.25 * (j_u + mj_u + 1) * (j_u + 2 + mj_u)
                        wl_int_tuple = (wavelength, Constant1D(rel_int_fine_structure * (rel_int / norm_const)))
                        sigma_minus_components.append(wl_int_tuple)

                    elif delta_mj == 0:
                        rel_int = (j_u + 1) ** 2 - mj_u ** 2
                        wl_int_tuple = (wavelength, Constant1D(rel_int_fine_structure * (rel_int / norm_const)))
                        pi_components.append(wl_int_tuple)

                elif delta_j == 0:
                    if delta_mj == 1:
                        rel_int = 0.25 * (j_u + mj_u) * (j_u + 1 - mj_u)
                        wl_int_tuple = (wavelength, Constant1D(rel_int_fine_structure * (rel_int / norm_const)))
                        sigma_plus_components.append(wl_int_tuple)

                    elif delta_mj == -1:
                        rel_int = 0.25 * (j_u - mj_u) * (j_u + 1 + mj_u)
                        wl_int_tuple = (wavelength, Constant1D(rel_int_fine_structure * (rel_int / norm_const)))
                        sigma_minus_components.append(wl_int_tuple)

                    elif delta_mj == 0:
                        rel_int = mj_u ** 2
                        wl_int_tuple = (wavelength, Constant1D(rel_int_fine_structure * (rel_int / norm_const)))
                        pi_components.append(wl_int_tuple)

    return pi_components, sigma_plus_components, sigma_minus_components





