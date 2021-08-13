import os
import yaml
import numpy as np
from scipy.constants import h, c, e, physical_constants
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))


def zeeman(bfield, view=0):
    """
    Zeeman-split line component wavelengths, relative intensities and polarisation states.

    Weak-field anomolous Zeeman effect. Assumes that ions are well described by L-S coupling.

    Based on 'specline.m' script by Scott Silburn

    :return:
    - wavelengths: list of wavelengths
    - relative_intensities: list of relative intensities.

    TODO: this script is unfinished, need to account for view_angle and polarisation effects
    """

    bfield_mag = np.abs(bfield)  # TODO
    mu_b = physical_constants['Bohr magneton in eV/T'][0]


    with open(os.path.join(dir_path, 'ciii.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    no_transitions = len(config)
    wls = []
    rel_ints = []

    # loop over each transition
    for ii in range(no_transitions):

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
                    rel_int = rel_int * (np.sin(view))**2
                else:
                    rel_int = rel_int * (1 + (np.cos(view))**2)

                rel_ints.append(rel_int * rel_int_fine_structure)

    # normalise the intensities
    const = np.sum(rel_ints)
    norm_rel_ints = rel_ints/const

    return wls, norm_rel_ints

if __name__ == '__main__':
    bfield = 0.
    wls, ris = zeeman(bfield, 45)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for wl, ri in zip(wls, ris):
        ax.plot([wl, wl, ], [0, ri],  )

    plt.show()






