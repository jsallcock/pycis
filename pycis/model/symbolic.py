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
# ----------------------------------------------------------------------------------------------------------------------
s0, s1, s2, s3 = symbols('s0 s1 s2 s3')
S_UNPOLARISED = Matrix([s0, 0, 0, 0])
S_GENERAL = Matrix([s0, s1, s2, s3])


# ----------------------------------------------------------------------------------------------------------------------
# MUELLER MATRICES
# ----------------------------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------------------------
def signal_1retarder_linear():
    phi = symbols('phi')
    mueller = polariser(0) * retarder(pi / 4, phi) * polariser(0)
    i_out = (mueller * S_UNPOLARISED)[0]
    print('1retarder_linear:')
    print(simplify(trigsimp(i_out)))
    print(' ')


def signal_1retarder_pixelated():
    phi, m = symbols('phi m')
    mueller = polariser(m * pi / 4) * qwp(pi / 2) * retarder(pi / 4, phi) * polariser(0)
    i_out = (mueller * S_UNPOLARISED)[0]
    print('1retarder_pixelated:')
    print(simplify(trigsimp(i_out)))
    print(' ')


def signal_2retarder_linear():
    phi_1, phi_2 = symbols('phi_1 phi_2')
    mueller = [
        polariser(0) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 4),
        polariser(0) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 8),
        polariser(pi / 8) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 8),
    ]
    labs = [
        '2-delay:',
        '3-delay:',
        '4-delay:',
    ]
    print('2retarder_linear:')
    for m, lab in zip(mueller, labs):
        i_out = (m * S_UNPOLARISED)[0]
        print(lab, simplify(trigsimp(i_out)))
    print(' ')


# ----------------------------------------------------------------------------------------------------------------------
# EXAMPLES - SPECTRO-POLARIMETRY
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    signal_1retarder_linear()
    signal_1retarder_pixelated()
    signal_2retarder_linear()
