"""
UKAEA cancelled their Matlab license, so I am testing out the SymPy package for Mueller calculus
"""
import numpy as np
from math import radians
from sympy import Matrix, sin,  cos, symbols, simplify, pi, trigsimp, init_printing, sqrt
init_printing()

def rot(rho):
    """ Frame rotation Mueller matrix. rho is rotation angle about x-axis
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
    """ Polariser Mueller matrix. rho is angle in radians of transmission axis about x-axis
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
    """ Retarder Mueller matrix. rho is angle in radians of fast axis about x-axis
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


s0, s1, s2, s3 = symbols('s0 s1 s2 s3')
phi = symbols('phi')
S_UNPOLARISED = Matrix([s0, 0, 0, 0])
S_GENERAL = Matrix([s0, s1, s2, s3])
# i = (polariser(0) * retarder(pi / 4, phi) * polariser(0) * S_GENERAL)[0]


phi_1, phi_2, phi_3, m, rho_1, rho_2 = symbols('phi_1 phi_2 phi_3 m rho_1 rho_2')
POL_1 = polariser(rho_1)
RET_1 = retarder(0, phi_1)
RET_2 = retarder(pi / 4, phi_2)
RET_3 = retarder(0, phi_3)
POL_2 = polariser(pi / 4)
print(trigsimp(POL_2 * RET_3 * RET_2 * RET_1 * POL_1 * S_UNPOLARISED)[0])


"""
triple-delay interferometer from P. Urlings' 2015 thesis
"""
# phi_6mm, phi_3mm, phi_4mm = symbols('phi_6mm phi_3mm phi_4mm')
# i_purlings = trigsimp(
#         polariser(0) *
#         retarder(pi / 8, pi) *
#         retarder(pi / 2, phi_6mm) *
#         retarder(0, phi_3mm) *
#         retarder(pi / 4, phi_4mm) *
#         polariser(pi / 8) *
#         S_UNPOLARISED
# )[0]

"""
Double-delay (basic)
"""
# phi_1, phi_2 = symbols('phi_1 phi_2')
# POL_1 = polariser(0)
# RET_1 = retarder(pi / 4, phi_1)
# HWP = retarder(pi / 8, pi)
# RET_2 = retarder(pi / 4, phi_2)
# POL_2 = polariser(0)
# i_2 = trigsimp(POL_1 * RET_1 * HWP * RET_2 * POL_2 * S_UNPOLARISED)[0]
# print(i_2)

"""
Double-delay (alt)
"""
# phi_1, phi_2, m, rho_1, rho_2 = symbols('phi_1 phi_2 m rho_1 rho_2')
# POL_1 = polariser(pi / 4)
# RET_1 = retarder(0, phi_1)
# RET_2 = retarder(pi / 4, phi_2)
# POL_2 = polariser(0)
# print(trigsimp(POL_2 * RET_2 * RET_1 * POL_1 * S_UNPOLARISED)[0])

# print(trigsimp(polariser(m*pi / 4) * retarder(pi / 2, pi / 2) * RET_2 * RET_1 * polariser(pi / 8) * S_UNPOLARISED)[0])  # triple delay pixelated
# print(trigsimp(POL_2 * RET_2 * RET_1 * polariser(pi / 8) * S_UNPOLARISED)[0])  # triple delay
# print(trigsimp(polariser(pi / 8) * RET_2 * RET_1 * polariser(pi / 8) * S_UNPOLARISED)[0])
# print(trigsimp(polariser(pi / 8 + m*pi / 4) * retarder(pi / 8 + pi / 2, pi / 2) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 8) * S_UNPOLARISED)[0])
# print(trigsimp(polariser(rho_2) * RET_2 * RET_1 * polariser(rho_1) * S_UNPOLARISED)[0])
# print(trigsimp(polariser(0) * retarder(rho_2, phi_2) * retarder(45 * pi / 180, phi_1) * polariser(0) * S_UNPOLARISED)[0])
# print('double delay')
# print(trigsimp((retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 4) * S_UNPOLARISED)))
# print(' ')
# print(polariser(0))
# print(polariser(m * pi / 4) * retarder(pi / 2, pi / 2))
# print(' ')
# print(' ')
# print('quad delay')
# into = trigsimp((retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 8) * S_UNPOLARISED))
# print(into)
# # print(trigsimp((polariser(m * pi / 4) * retarder(pi / 8, pi) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * polariser(pi / 8) * S_UNPOLARISED)))
# print(' ')
# print(polariser(pi / 8))
# print(polariser(m * pi / 4) * retarder(pi / 2, pi / 2))
#
# mat = Matrix([[0.5, 0.25 * sqrt(2) * cos(m * pi / 2), 0.25*sqrt(2), -0.25 * sqrt(2) * sin(m * pi / 2)], [0, 0, 0, 0], [0, 0, 0, 0, ], [0, 0, 0, 0, ]])
# print(mat)
# print(simplify(mat * into)[0])
# print(' ')
# print(polariser(m * pi / 4) * retarder(pi / 2, pi / 2))
# print(polariser(m * pi / 4) * retarder(pi / 2, rho_1))
# print(polariser(m * pi / 4) * retarder(pi / 8 + pi / 2, pi / 2))
"""
Double-delay (pixelated)
"""
# phi_1, phi_2, m = symbols('phi_1 phi_2 m')
# POL_1 = polariser(0)
# RET_1 = retarder(pi / 4, phi_1)
# HWP = retarder(pi / 8, pi)
# RET_2 = retarder(pi / 4, phi_2)
# QWP = retarder(pi / 2, pi / 2)
# POL_2 = polariser(m * pi / 4)
# i_2_pix = trigsimp(POL_2 * QWP * RET_2 * HWP * RET_1 * POL_1 * S_UNPOLARISED)[0]
# print(i_2_pix)

"""
triple-delay interferometer from P. Urlings' 2015 thesis
"""
# phi_1, phi_2, phi_3 = symbols('phi_1 phi_2 phi_3')
#
# POL_1 = polariser(0)
# RET_1 = retarder(pi / 4, phi_1)
# HWP = retarder(pi / 8, pi)
# RET_2 = retarder(0, phi_2)
# RET_3 = retarder(pi / 4, phi_3)
# POL_2 = polariser(0)
# i_3 = trigsimp(POL_1 * RET_1 * RET_2 * HWP * RET_3 * POL_2 * S_UNPOLARISED)[0]
# print(i_3)
