import numpy as np
from math import radians
from sympy import Matrix, sin,  cos, symbols, simplify, pi, trigsimp, exp, I, sqrt, re, sqrt_mod, im


def rot(rho):
    """ Frame rotation Jones matrix. rho is rotation angle about x-axis
    """
    rot = Matrix(
        [
            [cos(rho), sin(rho)],
            [-sin(rho), cos(rho)],
        ]
    )
    return rot


def polariser(rho):
    """ Polariser Mueller matrix. rho is angle in radians of transmission axis about x-axis
    """
    polariser = Matrix(
        [
            [1, 0],
            [0, 0],
        ]

    )
    return rot(-rho) * polariser * rot(rho)


def retarder(rho, phi):
    """ Retarder Mueller matrix. rho is angle in radians of fast axis about x-axis
    """
    retarder = Matrix(
        [
            [exp(I * phi), 0],
            [0, 1],
        ]
    )
    return rot(-rho) * retarder * rot(rho)


def get_I(E):
    return simplify((E.norm() ** 2).rewrite(cos))

S_0 = Matrix([1, 0])
S_45 = 1 / sqrt(2) * Matrix([1, 1])
phi_1, phi_2, m = symbols('phi_1 phi_2 m', real=True, positive=True, )

I_out_1 = get_I(polariser(0) * retarder(pi / 4, phi_1) * S_0)
I_out_2 = get_I(polariser(0) * retarder(pi / 4, phi_2) * retarder(0, phi_1) * S_45)

print(I_out_1)
print(I_out_2)

# exploring
# print(retarder(0, phi_1) * S_45)
print(get_I(retarder(pi / 4, phi_2) * retarder(0, phi_1) * S_45))
print(retarder(pi / 4, phi_2) * retarder(0, phi_1) * S_45)




