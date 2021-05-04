import copy

import xarray as xr

DWL = 1.e-10
sellmeier_coefs_source_defaults = {
    'a-BBO': 'agoptics',
    'b-BBO': 'eimerl',
    'calcite': 'ghosh',
    'YVO': 'shi',
    'lithium_niobate': 'zelmon',
}

def get_refractive_indices(wavelength, material, sellmeier_coefs_source=None, sellmeier_coefs=None, ):
    """
    Calculate the extraordinary and ordinary refractive indices as a function of wavelength.

    :param wavelength: Wavelength in m.
    :type wavelength: float, numpy.ndarray, xarray.DataArray
    :param str material: Set crystal material.
    :param str sellmeier_coefs_source: Specify which source to use for the Sellmeier coefficients that describe the
    dispersion. If not specified, defaults for each material are set by sellmeier_coefs_source_defaults in
    pycis.model.dispersion.
    :param dict sellmeier_coefs: Manually set the coefficients that describe the material dispersion
    via the Sellmeier equation. Dictionary must have keys 'Ae', 'Be', 'Ce', 'De', 'Ao', 'Bo', 'Co', 'Do'.
    :return:
    """

    if all([arg is not None for arg in [sellmeier_coefs_source, sellmeier_coefs]]):
        raise ValueError('pycis: arguments not understood')

    if sellmeier_coefs is not None:
        form = 1
    else:
        sellmeier_coefs = get_sellmeier_coefs(material, sellmeier_coefs_source)
    return sellmeier_eqn(wavelength * 1e6, sellmeier_coefs, )


def get_kappa(wavelength, material, **kwargs):
    """
    Calculate kappa, the dimensionless parameter that gives a first-order account of material dispersion.

    :param wavelength: Wavelength(s) in units m.
    :type wavelength: float, np.array, xr.DataArray
    :param str material: Specifies the material.
    :return: Kappa
    """

    ne, no = get_refractive_indices(wavelength, material, **kwargs)
    ne_p1, no_p1 = get_refractive_indices(wavelength + DWL, material, **kwargs)
    ne_m1, no_m1 = get_refractive_indices(wavelength - DWL, material, **kwargs)

    biref = ne - no
    biref_p1 = ne_p1 - no_p1
    biref_m1 = ne_m1 - no_m1
    biref_deriv = (biref_p1 - biref_m1) / (2 * DWL)

    return 1 - (wavelength / biref) * biref_deriv


def sellmeier_eqn(wl, c, ):
    """
    Given a set of Sellmeier coefficients, calculate the extraordinary and ordinary refractive indices as a function of
    wavelength.

    :param array-like wl: Wavelength(s) in microns.
    :param dict c: Dictionary of Sellmeier coefficients.
    :param int form: Which form of Sellmeier equation to use.
    :return: (n_e, n_o) tuple containing extraordinary and ordinary refractive indices respectively as a function of
    wavlength.
    """
    ts = ['e', 'o']
    if len(c) == 8:
        return [(c['A'+t] + (c['B'+t] / (wl ** 2 + c['C'+t])) + (c['D'+t] * wl ** 2)) ** 0.5 for t in ts]
    elif len(c) == 10:
        return [(c['A'+t] + (c['B'+t] / (wl ** 2 + c['C'+t])) + (c['D'+t] / (wl ** 2 + c['E'+t]))) ** 0.5 for t in ts]
    elif len(c) == 12:
        return [((c['A'+t] * wl ** 2 / (wl ** 2 - c['B'+t])) +
                (c['C'+t] * wl ** 2 / (wl ** 2 - c['D'+t])) +
                (c['E'+t] * wl ** 2 / (wl ** 2 - c['F'+t])) + 1) ** 0.5 for t in ts]
    else:
        raise NotImplementedError


def get_sellmeier_coefs(material, sellmeier_coefs_source=None):
    """
    :param sellmeier_coefs_source:
    :return:
    """
    if sellmeier_coefs_source is None:
        sellmeier_coefs_source = sellmeier_coefs_source_defaults[material]
    return copy.deepcopy(sellmeier_coefs_all[sellmeier_coefs_source]['sellmeier_coefs'])


sellmeier_coefs_all = {
    'kato1986':
        {'material': 'b-BBO',
         'sellmeier_coefs':
             {'Ae': 2.3753,
              'Be': 0.01224,
              'Ce': -0.01667,
              'De': -0.01516,
              'Ao': 2.7359,
              'Bo': 0.01878,
              'Co': -0.01822,
              'Do': -0.01354,
              },
         },

    'kato2010':
        {'material': 'b-BBO',
         'sellmeier_coefs':
             {'Ae': 3.33469,
              'Be': 0.01237,
              'Ce': -0.01647,
              'De': 79.0672,
              'Ee': -82.2919,
              'Ao': 3.63357,
              'Bo': 0.018778,
              'Co': -0.01822,
              'Do': 60.9129,
              'Eo': -67.8505,
              },
         },

    'eimerl':
        {'material': 'b-BBO',
         'sellmeier_coefs':
             {'Ae': 2.3730,
              'Be': 0.0128,
              'Ce': -0.0156,
              'De': -0.0044,
              'Ao': 2.7405,
              'Bo': 0.0184,
              'Co': -0.0179,
              'Do': -0.0155,
              },
         },

    'kim':
        {'material': 'a-BBO',
         'sellmeier_coefs':
             {'Ae': 2.37153,
              'Be': 0.01224,
              'Ce': -0.01667,
              'De': -0.01516,
              'Ao': 2.7471,
              'Bo': 0.01878,
              'Co': -0.01822,
              'Do': -0.01354,
              },
         },

    'agoptics':
        {'material': 'a-BBO',
         'sellmeier_coefs':
             {'Ae': 2.3753,
              'Be': 0.01224,
              'Ce': -0.01667,
              'De': -0.01516,
              'Ao': 2.7471,
              'Bo': 0.01878,
              'Co': -0.01822,
              'Do': -0.01354,
              },
         },

    'newlightphotonics':
        {'material': 'a-BBO',
         'sellmeier_coefs':
             {'Ae': 2.31197,
              'Be': 0.01184,
              'Ce': -0.01607,
              'De': -0.00400,
              'Ao': 2.67579,
              'Bo': 0.02099,
              'Co': -0.00470,
              'Do': -0.00528,
              },
         },

    'ghosh':
        {'material': 'calcite',
         'sellmeier_coefs':
             {'Ae': 1.35859695,
              'Be': 0.82427830,
              'Ce': 1.06689543e-2,
              'De': 0.14429128,
              'Ee': 120,
              'Ao': 1.73358749,
              'Bo': 0.96464345,
              'Co': 1.94325203e-2,
              'Do': 1.82831454,
              'Eo': 120,
              }
         },
    'shi':
        {'material': 'YVO',
         'sellmeier_coefs':
             {'Ae': 4.607200,
              'Be': 0.108087,
              'Ce': 0.052495,
              'De': 0.014305,
              'Ao': 3.778790,
              'Bo': 0.070479,
              'Co': 0.045731,
              'Do': 0.009701,
              },
         },

    'zelmon':  # D.E. Zelmon, D. L. Small, J. Opt. Soc. Am. B/Vol. 14, No. 12/December 1997
        {'material': 'lithium_niobate',
         'sellmeier_coefs':
             {'Ae': 2.9804,
              'Be': 0.02047,
              'Ce': 0.5981,
              'De': 0.0666,
              'Ee': 8.9543,
              'Fe': 416.08,
              'Ao': 2.6734,
              'Bo': 0.01764,
              'Co': 1.2290,
              'Do': 0.05914,
              'Eo': 12.614,
              'Fo': 474.6,
              },
         },
}
