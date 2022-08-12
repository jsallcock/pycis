"""
Basic 'symbolic computation' implementation of Mueller calculus framework, used to derive the equations for
interference fringe patterns.

Currently this module exists by itself, not interacting with the tools for synthetic image generation.
"""
import numpy as np
from math import radians
from sympy import Matrix, sin,  cos, symbols, simplify, pi, trigsimp, init_printing, sqrt
init_printing()


# ----------------------------------------------------------------------------------------------------------------------
# STOKES VECTORS
s0, s1, s2, s3 = symbols('s0 s1 s2 s3')
S_UNPOLARISED = Matrix([s0, 0, 0, 0])
S_GENERAL = Matrix([s0, s1, s2, s3])


# ----------------------------------------------------------------------------------------------------------------------
# MUELLER MATRICES
def rot(rho):
    """
    Frame rotation matrix

    :param float rho: rotation angle about x-axis
    :return: 4x4 rotation matrix
    """
    rot = Matrix(
        [
            [1, 0, 0, 0],
            [0, cos(2 * rho), sin(2 * rho), 0],
            [0, -sin(2 * rho), cos(2 * rho), 0],
            [0, 0, 0, 1],
        ]
    )
    return rot


def polariser(rho):
    """
    Polariser Mueller matrix

    :param float rho: angle in radians of polariser transmission axis about x-axis.
    :return:
    """
    polariser = Matrix(
        [
            [0.5, 0.5, 0, 0],
            [0.5, 0.5, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    return rot(-rho) * polariser * rot(rho)


def retarder(rho, phi):
    """
    Retarder Mueller matrix

    :param float rho: angle in radians of retarder fast axis about x-axis.
    :param float phi: angle in radians of imparted retardance.
    :return: Mueller matrix.
    """
    retarder = Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cos(phi), sin(phi)],
            [0, 0, -sin(phi), cos(phi)],
        ]
    )
    return rot(-rho) * retarder * rot(rho)


def qwp(rho):
    """
    Quarter-wave plate Mueller matrix

    :param float rho: angle in radians of retarder fast axis about x-axis.
    :return: Mueller matrix
    """
    return retarder(rho, pi / 2)


# ----------------------------------------------------------------------------------------------------------------------
# EXAMPLES - SPECTROSCOPY
def spec_1retarder_linear():
    phi, rho = symbols('phi rho', real=True, nonnegative=True)
    mueller = polariser(rho) * retarder(rho + pi / 4, phi) * polariser(rho)
    i_out = trigsimp((mueller * S_UNPOLARISED)[0])
    i_out_gen = trigsimp((mueller * S_GENERAL)[0])
    print('spec_1retarder_linear:')
    print(i_out)
    print(i_out_gen)
    print(' ')


def spec_1retarder_pixelated():
    phi, m, rho = symbols('phi m rho', real=True, nonnegative=True)
    mueller = polariser(m * pi / 4) * qwp(pi / 2 + rho) * retarder(pi / 4 + rho, phi) * polariser(0 + rho)
    i_out = trigsimp((mueller * S_UNPOLARISED)[0])
    print('spec_1retarder_pixelated:')
    print(trigsimp(i_out))
    print(' ')


def spec_2retarder_linear():
    phi_1, phi_2, rho_1, rho_2 = symbols('phi_1 phi_2 rho_1 rho_2')
    mueller = [
        polariser(0) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 4),
        polariser(0) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 8),
        polariser(pi / 8) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 8),
        # polariser(0) * retarder(rho_2, phi_2) * retarder(rho_1, phi_1) * polariser(0),
    ]
    labs = [
        '2-delay:',
        '3-delay:',
        '4-delay:',
        # 'gen:'
    ]
    print('spec_2retarder_linear:')
    for m, lab in zip(mueller, labs):
        i_out = (m * S_UNPOLARISED)[0]
        print(lab, simplify(trigsimp(i_out)))
    print(' ')



def spec_2retarder_linear_hwp():
    phi_1, phi_2, rho_1, rho_2 = symbols('phi_1 phi_2 rho_1 rho_2')
    mueller = [
        # polariser(0) * retarder(3*pi / 4, phi_2) * retarder(pi / 8, pi) * retarder(pi / 4, phi_1) * polariser(0),
        polariser(pi / 4) * retarder(pi / 2, phi_2) * retarder(pi / 8, pi) * retarder(0, phi_1) * polariser(pi / 4),
        # polariser(0) * retarder(3 * pi / 4, phi_2) * retarder(pi / 8, pi) * retarder(pi / 4, phi_1) * polariser(pi / 8),
        polariser(pi / 4) * retarder(pi / 2, phi_2) * retarder(pi / 8, pi) * retarder(0, phi_1) * polariser(pi / 8),
        polariser(pi / 8) * retarder(3 * pi / 4, phi_2) * retarder(pi / 8, pi) * retarder(pi / 4, phi_1) * polariser(pi / 8),
    ]
    labs = [
        '2-delay:',
        '3-delay:',
        '4-delay:',
        # 'gen:'
    ]
    print('spec_2retarder_linear:')
    for m, lab in zip(mueller, labs):
        i_out = (m * S_UNPOLARISED)[0]
        print(lab, simplify(trigsimp(i_out)))
    print(' ')


def spec_3retarder_linear():
    phi_1, phi_2, phi_3 = symbols('phi_1 phi_2 phi_3')
    mueller = [
        polariser(0) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * retarder(pi / 4, phi_3) * polariser(0),
        # polariser(0) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 8),
        # polariser(pi / 8) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 8),
    ]
    labs = [
        'attempt:',
        # '3-delay:',
        # '4-delay:',
    ]
    print('spec_3retarder_linear:')
    for m, lab in zip(mueller, labs):
        i_out = (m * S_UNPOLARISED)[0]
        print(lab, simplify(trigsimp(i_out)))
    print(' ')


def spec_2retarder_pixelated():
    phi_1, phi_2, m = symbols('phi_1 phi_2 m')
    mueller = [
        polariser(m * pi / 4) * qwp(pi / 2) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 4),
        polariser(m * pi / 4) * qwp(pi / 2) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 8),
    ]
    labs = [
        '2-delay:',
        '3-delay:',
    ]
    print('spec_2retarder_pixelated:')
    for m, lab in zip(mueller, labs):
        i_out = (m * S_UNPOLARISED)[0]
        print(lab, simplify(trigsimp(i_out)))
    print(' ')


def spec_3retarder_pixelated():
    phi_1, phi_2, phi_3, m = symbols('phi_1 phi_2 phi_3 m')
    mueller = [
        polariser(m * pi / 4) * qwp(pi / 2) * retarder(pi / 4, phi_3) * retarder(0, phi_2) * retarder(pi / 4, phi_1) * polariser(0),
        polariser(m * pi / 4) * qwp(pi / 2) * retarder(pi / 4, phi_3) * retarder(0, phi_2) * retarder(pi / 4, phi_1) * polariser(pi / 2 ),
    ]
    labs = [
        '6-delay:',
        'testing',
    ]
    print('spec_3retarder_pixelated:')
    for m, lab in zip(mueller, labs):
        i_out = (m * S_UNPOLARISED)[0]
        print(lab, simplify(trigsimp(i_out)))
    print(' ')

# ----------------------------------------------------------------------------------------------------------------------
# EXAMPLES - SPECTRO-POLARIMETRY
def specpol_2retarder_linear():
    """
    generalised full-stokes polarimeter
    """
    phi = symbols('phi')
    mueller = polariser(0) * retarder(pi / 4, phi) * retarder(0, phi)
    i_out = (mueller * S_GENERAL)[0]
    print('specpol_1retarder_linear:')
    print(simplify(trigsimp(i_out)))
    print(' ')


if __name__ == '__main__':
    # spec_2retarder_linear()
    spec_2retarder_pixelated()
    # spec_3retarder_pixelated()
    # spec_3retarder_linear()
    # spec_1retarder_linear()
    # spec_1retarder_pixelated()
    # spec_2retarder_pixelated()
    # spec_2retarder_linear()
    # specpol_2retarder_linear()