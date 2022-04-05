import subprocess
import numpy as np
import matplotlib.pyplot as plt


def adas603_get_zeeman(pol, obsangle, bvalue, findex):
    """
    get wavelengths and relative intensities of Zeeman-split line components.

    This is a dirty / untested port of ADAS603_get_zeeman.pro. It just calls an executable in the ADAS repo.
    This should be valid for weak-field, intermediate-field and strong-field Zeeman splitting, but I currently have no
    way of testing the output.

    Original IDL can be found on Freia at /home/adas/idl/adas6xx/adas603/adas603_get_zeeman.pro
    :param int pol: 0 = NONE,
                    1 = PI+SIGMA,
                    2 = PI,
                    3 = SIGMA,
                    4 = SIGMA+,
                    5 = SIGMA-
    :param float obsangle: Observation angle (degrees -- based on the output).
    :param float bvalue: Magnetic field strength (T).
    :param int findex: Feature index (see interactive ADAS603 GUI for full list)
    :return: list of wavelengths (in m), list of corresponding relative intensities (normalised)
    """

    cmd = "/home/adas/bin64/components603"
    proc1 = subprocess.Popen(
        'echo ' + str(pol) + ' ' + str(obsangle) + ' ' + str(bvalue) + ' ' + str(findex),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    proc2 = subprocess.Popen(
        cmd,
        stdin=proc1.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    out = proc2.communicate()[0].decode('utf-8').split('\n')
    nlines = int(out[0])
    wls = []
    rel_ints = []
    for ii in range(nlines):
        elements = out[ii + 1].split('#')
        wls.append(float(elements[2]) * 1e-10)
        rel_ints.append(float(elements[3]))

    return wls, rel_ints


def plot():
    """ just trying to make sense of above """

    findex = 28
    obsangle = 3.1415 / 2
    bvalue = 2.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for pol in [0, 1]:
        wls, rel_ints = adas603_get_zeeman(pol, obsangle, bvalue, findex)
        col = 'C' + str(pol)

        for ii_plot, (wl, rel_int) in enumerate(zip(wls, rel_ints)):
            if ii_plot == 0:
                label = str(pol)
            else:
                label = None
            ax.plot([wl, wl, ], [0, rel_int], lw=2, marker='x', color=col, label=label)

        print('pol =', str(pol), )
        print('rel_ints.sum() = ', np.array(rel_ints).sum())
        print('----------')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
    # wls, rel_ints = adas603_get_zeeman(1, 0., 1., 28)
    # print(np.array(rel_ints).sum())
