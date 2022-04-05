import copy
import numpy as np
import matplotlib.pyplot as plt
import pycis
from matplotlib.gridspec import GridSpec
from scipy.constants import c

DWL = 1.e-10
sellmeier_coefs_source_defaults = {
    'a-BBO': 'agoptics',
    'b-BBO': 'eimerl',
    'calcite': 'ghosh',
    'YVO': 'shi',
    'lithium_niobate': 'zelmon',
}


def get_refractive_indices(wavelength, material='a-BBO', sellmeier_coefs_source=None, sellmeier_coefs=None, ):
    """
    Calculate the extraordinary and ordinary refractive indices as a function of wavelength

    :param wavelength: Wavelength in m.
    :type wavelength: float, numpy.ndarray, xarray.DataArray

    :param str material: Set crystal material.

    :param str sellmeier_coefs_source: \
        Specify which source to use for the Sellmeier coefficients that describe the
        dispersion. If not specified, defaults for each material are set by sellmeier_coefs_source_defaults in
        pycis.model.dispersion.

    :param dict sellmeier_coefs: \
        Manually set the coefficients that describe the material dispersion
        via the Sellmeier equation. Dictionary must have keys 'Ae', 'Be', 'Ce', 'De', 'Ao', 'Bo', 'Co' and 'Do'.

    :return: (ne, no) tuple containing extraordinary and ordinary refractive indices respectively. type(ne) = type(no)
        = type(wavelength).

    """
    if all([arg is not None for arg in [sellmeier_coefs_source, sellmeier_coefs]]):
        raise ValueError('pycis: arguments not understood')

    if sellmeier_coefs is not None:
        form = 1
    else:
        sellmeier_coefs = get_sellmeier_coefs(material, sellmeier_coefs_source)
    return sellmeier_eqn(wavelength * 1e6, sellmeier_coefs, )


def get_kappa(wavelength, **kwargs):
    """
    Calculate kappa, the dimensionless parameter that gives a first-order account of material dispersion, as a function
    of wavelength

    :param wavelength: Wavelength in m.
    :type wavelength: float, numpy.ndarray, xarray.DataArray

    :param str material: Set crystal material.

    :param str sellmeier_coefs_source: \
        Specify which source to use for the Sellmeier coefficients that describe the
        dispersion. If not specified, defaults for each material are set by sellmeier_coefs_source_defaults in
        pycis.model.dispersion.

    :param dict sellmeier_coefs: \
        Manually set the coefficients that describe the material dispersion
        via the Sellmeier equation. Dictionary must have keys 'Ae', 'Be', 'Ce', 'De', 'Ao', 'Bo', 'Co' and 'Do'.

    :return: kappa. type(kappa) = type(wavelength)

    """
    ne, no = get_refractive_indices(wavelength, **kwargs)
    ne_p1, no_p1 = get_refractive_indices(wavelength + DWL, **kwargs)
    ne_m1, no_m1 = get_refractive_indices(wavelength - DWL, **kwargs)

    biref = ne - no
    biref_p1 = ne_p1 - no_p1
    biref_m1 = ne_m1 - no_m1
    biref_deriv = (biref_p1 - biref_m1) / (2 * DWL)

    return 1 - (wavelength / biref) * biref_deriv


def sellmeier_eqn(wl, c, ):
    """
    Given a set of Sellmeier coefficients, calculate the extraordinary and ordinary refractive indices as a function of
    wavelength

    :param wl: Wavelength in microns.
    :type wl: float, numpy.ndarray, xarray.DataArray

    :param dict c: \
        Coefficients that describe material dispersion via the Sellmeier equation. Dictionary must have keys 'Ae', 'Be',
         'Ce', ... and 'Ao', 'Bo', 'Co', ... The form of Sellmeier equation used is determined by len(c).

    :return: (ne, no) tuple containing extraordinary and ordinary refractive indices respectively as a function of
        wavelength.

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
    Return the requested set of Sellmeier coefficients

    :param str material:

    :param str sellmeier_coefs_source: \
        Specify which source to use for the Sellmeier coefficients that describe the
        dispersion. If not specified, defaults for each material are set by sellmeier_coefs_source_defaults in
        pycis.model.dispersion.

    :return: dict containing the Sellmeier coefficients.
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

    'appel':
            {'material': 'a-BBO',
             'sellmeier_coefs':
                 {'Ae': 2.3174,
                  'Be': 0.01224,
                  'Ce': -0.01667,
                  'De': -0.01516,
                  'Ao': 2.7471,
                  'Bo': 0.01878,
                  'Co': -0.01822,
                  'Do': -0.01354,
                  },
             },

    'shi':
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
    # 'shi':
    #     {'material': 'YVO',
    #      'sellmeier_coefs':
    #          {'Ae': 4.607200,
    #           'Be': 0.108087,
    #           'Ce': 0.052495,
    #           'De': 0.014305,
    #           'Ao': 3.778790,
    #           'Bo': 0.070479,
    #           'Co': 0.045731,
    #           'Do': 0.009701,
    #           },
    #      },

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


def plot_dispersion():
    fig = plt.figure(figsize=(6.104, 6.9))
    gs = GridSpec(1, 3, figure=fig, wspace=0.4)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    ax1, ax2, ax3 = axes
    wl = np.linspace(400e-9, 700e-9, 500)
    wl_nm = wl * 1e9

    materials = ['a-BBO',
                 'a-BBO',
                 'a-BBO',
                 # 'a-BBO',
                 'a-BBO',
                 'b-BBO',
                 'b-BBO',
                 'b-BBO',
                 ]
    sources = ['agoptics',
               'kim',
               'newlightphotonics',
               # 'appel',
               'shi',
               'eimerl',
               'kato1986',
               'kato2010',
               ]
    labels = ['$\\alpha$/1',
              '$\\alpha$/2',
              '$\\alpha$/3',
              # '$\\alpha$/4',
              '$\\alpha$/5',
              '$\\beta$/1',
              '$\\beta$/2',
              '$\\beta$/3',
              ]

    for ii, (material, source, label) in enumerate(zip(materials, sources, labels)):
        color = 'C' + str(ii)
        ne, no = pycis.get_refractive_indices(wl, material=material, sellmeier_coefs_source=source)
        b = ne - no
        ax1.plot(wl_nm, no, color=color, label=label)
        ax2.plot(wl_nm, ne, color=color, label=label)
        ax3.plot(wl_nm, b, color=color, label=label)

        if ii == 0:
            no_l, no_u = no.min(), no.max()
            ne_l, ne_u = ne.min(), ne.max()
            b_l, b_u = b.min(), b.max()
        else:
            if no.min() < no_l:
                no_l = no.min()
            if no.max() > no_u:
                no_u = no.max()
            if ne.min() < ne_l:
                ne_l = ne.min()
            if ne.max() > ne_u:
                ne_u = ne.max()
            if b.min() < b_l:
                b_l = b.min()
            if b.max() > b_u:
                b_u = b.max()

    lims = [[no_l, no_u],
            [ne_l, ne_u],
            [b_l, b_u],
            ]
    titles = ['(a) $n_\\mathrm{O}(\\lambda)$',
              '(b) $n_\\mathrm{E}(\\lambda)$',
              '(c) $B(\\lambda)\\equiv{}n_\\mathrm{E}(\\lambda)-n_\\mathrm{O}(\\lambda)$', ]
    for ax, title, lim in zip(axes, titles, lims):
        ax.set_xlim(wl_nm.min(), wl_nm.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.6)
        ax.set_xlabel('$\\lambda$ (nm)')
        ax.set_title(title)
        ax.set_ylim(lim)

    txts = ['$\\alpha$-BBO', '$\\beta$-BBO']
    xys = [[550, -0.1216], [500, -0.119]]
    rots = [55, 60]
    for txt, xy, rot in zip(txts, xys, rots):
        ax3.annotate(txt, xy, xycoords='data', rotation=rot, rotation_mode='default')

    # ax3.set_ylabel('Birefringence')
    leg1 = ax1.legend(ncol=1, fontsize=10, title='Sellmeier\ncoefficients:')
    # fpath_fig = os.path.join(dir, 'sellmeier_dispersion.pdf')
    # fig.savefig(fpath_fig, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    plot_dispersion()
