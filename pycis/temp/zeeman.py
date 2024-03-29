import os
import yaml
import numpy as np
from scipy.constants import h, c, e, physical_constants
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))


def zeeman(bfield, view=0, stokes=False):
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

    bfield_mag = np.abs(bfield)  # TODO
    mu_b = physical_constants['Bohr magneton in eV/T'][0]

    # convert view angle to radians
    view = np.pi * view / 180


    with open(os.path.join(dir_path, 'ciii.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    no_transitions = len(config)
    wls = []
    norm_rel_ints = []
    norm_stokes_vector = []

    # loop over each transition
    for ii in range(no_transitions):
        print(ii)

        rel_ints = []
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
                energy_u_ii = energy_u + (mj_u * mu_b * bfield_mag * g_u)  # [eV]
                energy_l_ii = energy_l + (mj_l * mu_b * bfield_mag * g_l)  # [eV]
                wls.append(h * c / ((energy_u_ii - energy_l_ii) * e))  # [m]

                # calculate relative line intensities, expressions taken from Table 2.1, p23, S.Silburn's thesis
                if delta_j == 1:
                    if delta_mj == 1:
                        rel_int = 0.25 * (j_u + mj_u) * (j_u - 1 + mj_u)
                    elif delta_mj == -1:
                        rel_int = 0.25 * (j_u - mj_u) * (j_u - 1 - mj_u)
                    elif delta_mj == 0:
                        rel_int = j_u ** 2 - mj_u ** 2

                elif delta_j == -1:
                    if delta_mj == 1:
                        rel_int = 0.25 * (j_u + 1 - mj_u) * (j_u + 2 - mj_u)
                    elif delta_mj == -1:
                        rel_int = 0.25 * (j_u + mj_u + 1) * (j_u + 2 + mj_u)
                    elif delta_mj == 0:
                        rel_int = (j_u + 1) ** 2 - mj_u ** 2

                elif delta_j == 0:
                    if delta_mj == 1:
                        rel_int = 0.25 * (j_u + mj_u) * (j_u + 1 - mj_u)
                    elif delta_mj == -1:
                        rel_int = 0.25 * (j_u - mj_u) * (j_u + 1 + mj_u)
                    elif delta_mj == 0:
                        rel_int = mj_u ** 2


                # account for view angle. Should this be before or after normalisation?
                if delta_mj == 0:
                    rel_int0 = rel_int * 0.5 * (np.sin(view))**2
                    rel_int1 = rel_int * 0.5 * (np.sin(view))**2
                    rel_int2 = 0
                    rel_int3 = 0

                if delta_mj == 1:
                    rel_int0 = rel_int * 0.25 * (1 + (np.cos(view))**2)
                    rel_int1 = rel_int * -0.25 * (np.sin(view))**2
                    rel_int2 = 0
                    rel_int3 = rel_int * 0.5 * (np.cos(view))

                if delta_mj == -1:
                    rel_int0 = rel_int * 0.25 * (1 + (np.cos(view)) ** 2)
                    rel_int1 = rel_int * -0.25 * (np.sin(view)) ** 2
                    rel_int2 = 0
                    rel_int3 = rel_int * -0.5 * (np.cos(view))

                rel_ints_stokes = [rel_int0, rel_int1, rel_int2, rel_int3]

                rel_ints.append(rel_ints_stokes)
        # normalise the intensities. Does this this for each peak in the triplet separately.
        # print(wls, rel_ints)
        const = np.sum(item[0] for item in rel_ints)
        for stokes_vector in rel_ints:
            norm_stokes_vector = []
            for component in stokes_vector:
                norm_stokes_vector.append((component/const) * rel_int_fine_structure)
            norm_rel_ints.append(norm_stokes_vector)

    if stokes == False:
        new_norm_rel_ints = []
        for vector in norm_rel_ints:
            new_norm_rel_ints.append(vector[0])
        return wls, new_norm_rel_ints

    else:
        return wls, norm_rel_ints


if __name__ == '__main__':
    bfield = 0
    wls, ris = zeeman(bfield, )
    print(np.array(ris).sum())

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for wl, ri in zip(wls, ris):
        print(wl)
        ax.plot([wl, wl, ], [0, ri],  )

    plt.show()






