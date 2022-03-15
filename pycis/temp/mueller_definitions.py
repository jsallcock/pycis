from sympy import Matrix, sin,  cos, symbols, simplify, pi, trigsimp


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

# e.g. basic birefringent interferometer signal:
print((polariser(0) * retarder(pi / 4, phi) * polariser(0) * S_UNPOLARISED)[0])
