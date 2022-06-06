import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr
from scipy.constants import c, k, e, physical_constants
import pycis
import time
root = os.path.dirname(__file__)


def plot_demod_test():
    """

    :return:
    """

    # OPTIONS
    overwrite_output_img = False
    calibrate = True
    wl0 = 434.0472e-9
    wl0_cal = wl0

    # FLIR BLACKFLY POLARISATION CAMERA
    bit_depth = 12
    sensor_format = (200, 2000)
    # sensor_format = (200, 500)
    pix_size = 3.45e-6
    qe = 0.35
    epercount = 0.46  # [e / count]
    cam_noise = 2.5
    cam = pycis.Camera(sensor_format, pix_size, bit_depth, qe, epercount, cam_noise, type='monochrome_polarised')

    # define imaging lens
    optics = [20.e-3, 105.e-3, 50.e-3, ]
    wp1 = pycis.UniaxialCrystal(orientation=0, thickness=8.e-3, cut_angle=45)
    wp2 = pycis.UniaxialCrystal(orientation=45, thickness=9.8e-3, cut_angle=0)
    qwp = pycis.QuarterWaveplate(orientation=90.)
    pol1 = pycis.LinearPolariser(orientation=22.5)
    interferometer = [pol1, wp1, wp2, qwp]
    inst = pycis.Instrument(camera=cam, optics=optics, interferometer=interferometer)

    # generate spectrum
    nu0 = c / wl0
    i0_in = 1.e4
    spec_max = 0.  # keep track for plotting

    # at these wavelengths, the expected interferometric delays are:
    delay1 = wp1.get_delay(wl0, 0, 0)
    delay2 = wp2.get_delay(wl0, 0, 0)
    delay_diff = abs(delay1 - delay2)
    delay_sum = delay1 + delay2
    delays = np.array([delay2, delay_sum, delay_diff])
    delays_103 = delays * 1e-3
    delay_axis = np.linspace(0, 1.5 * delays.max(), 1000)  # for plotting predicted contrast profile.
    delay_axis = xr.DataArray(delay_axis, dims=('delay', ), coords=(delay_axis, ), )
    delay_axis_103 = delay_axis * 1e-3

    temps_ev = np.array([0.5, 1, 5])  # eV
    flows_kms = np.array([0., 15, 30])  # km / s
    cols = ['C0', 'C1', 'C2']

    # setup figure
    fig = plt.figure(figsize=(6.104, 6.75))
    gs_o_ud = GridSpec(3, 1, figure=fig, hspace=0.62, height_ratios=[0.75, 0.8, 1])
    gs_i_lru = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_o_ud[0], wspace=0.3, width_ratios=[2, 1])
    gs_i_lrd = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_o_ud[2], wspace=0.3, width_ratios=[5, 5,])
    gs_i_ud = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_o_ud[1], height_ratios=[1, 1, 1])
    ax_spec = fig.add_subplot(gs_i_lru[0])
    ax_cont = fig.add_subplot(gs_i_lrd[0])
    ax_ph = fig.add_subplot(gs_i_lrd[1])

    ax1, ax2, ax3 = [fig.add_subplot(gs_i_ud[iii]) for iii in range(3)]
    axes = [ax1, ax2, ax3]
    i_max = 0

    # GENERATE SYNTHETIC CALIBRATION IMAGE
    temp_ev_cal = 0.0001
    x, y = cam.get_pixel_position()  # camera method returns the camera's pixel x and y positions as DataArrays
    spectrum_cal = pycis.model.get_doppler_broadened_singlet(temp_ev_cal, wl0_cal, 2, 0, domain='wavelength', nbins=30)*i0_in
    spectrum_cal = xr.broadcast(spectrum_cal, x, y, )[0]

    sta = time.time()
    igram_cal = inst.capture(spectrum_cal, clean=True, )

    print('MAX =', igram_cal.max())
    end = time.time()
    print(end - sta, ' seconds')

    # DEMODULATE
    fringe_freq_cal = inst.get_fringe_frequency(wl0_cal)
    dc, phase_cal, contrast_cal = pycis.demod_triple_delay_pixelated(igram_cal, fringe_freq_cal)

    pm = cam.get_pixelated_phase_mask()
    sp = igram_cal * np.exp(1j*pm)
    fft = pycis.analysis.fft2_im(sp)
    fft = np.log10(abs(fft)**2)

    window_pm = pycis.analysis.make_carrier_window(fft, fringe_freq_cal, sign='pm')
    window_lowpass = pycis.analysis.make_lowpass_window(fft, fringe_freq_cal)

    plt.figure()
    fft.plot(x='freq_x', y='freq_y')
    plt.vlines(fringe_freq_cal[0], -100000, 100000)

    plt.figure()
    windowed_fft = fft*window_pm*window_lowpass
    windowed_fft.plot(x='freq_x', y='freq_y')

    plt.figure()
    phase_cal[0].plot(x='x', y='y')

    phase_c2_cal, phase_sum_cal, phase_diff_cal = phase_cal
    contrast_c2_cal, contrast_sum_cal, contrast_diff_cal = contrast_cal

    for iii, (temp_ev, flow_kms, col, ax, ) in enumerate(zip(temps_ev, flows_kms, cols, axes, )):
        temp_k = temp_ev * e / k
        vth = np.sqrt(2 * k * temp_k / physical_constants['deuteron mass'][0]) #eq. 5.1.4
        std_nu = np.sqrt((2 * nu0 ** 2 * vth ** 2) / c ** 2)
        delta_lambda = wl0 * flow_kms / (c / 1e3)
        wl0_shifted = wl0 + delta_lambda

        kappa0 = pycis.model.get_kappa(wl0_shifted, material='a-BBO', sellmeier_coefs_source='agoptics', )

        spectrum = pycis.model.get_doppler_broadened_singlet(temp_ev, wl0, 2, flow_kms, domain='wavelength', nbins=40)*i0_in
        wl_delta_nm = (spectrum.wavelength - wl0) * 1e9

        if float(spectrum.max()) > spec_max:
            spec_max = float(spectrum.max())
        ax_spec.plot(wl_delta_nm.values, spectrum.data, color=col)
        x, y = cam.get_pixel_position()
        spectrum = xr.broadcast(spectrum, x, y, )[0]

        cont_analytic = np.exp(-2 * np.pi ** 2 * (kappa0 * delay_axis / (2 * np.pi * nu0)) ** 2 * std_nu ** 2)
        phase_shift_analytic = kappa0 * delay_axis * delta_lambda / wl0

        # generate synthetic image
        sta = time.time()
        igram = inst.capture(spectrum, clean=True)
        end = time.time()
        print(end - sta, ' seconds')

        # DEMODULATE
        img = igram.values

        fringe_freq = inst.get_fringe_frequency(wl0)
        dc, phase_d, contrast_d = pycis.demod_triple_delay_pixelated(igram, fringe_freq)

        phase_c2_d,  phase_sum_d, phase_diff_d = phase_d
        contrast_c2_d, contrast_sum_d, contrast_diff_d = contrast_d

        # plt.figure()
        # phase_c2_d.plot(x='x', y='y')

        if calibrate:
            phase_c2_d = pycis.wrap(phase_c2_d - phase_c2_cal)
            phase_sum_d = pycis.wrap(phase_sum_d - phase_sum_cal)
            phase_diff_d = pycis.wrap(phase_diff_d - phase_diff_cal)

            contrast_c2_d /= contrast_c2_cal
            contrast_sum_d /= contrast_sum_cal
            contrast_diff_d /= contrast_diff_cal

        if iii == 0:
            i_max = img.max()
            print('IMG VAL MAX:', i_max)
        ax.imshow(img[int(4.5 * sensor_format[0] / 6):, int(6. * sensor_format[1] / 10):], 'gray', vmin=0, vmax=i_max,
                  rasterized=True)

        # roi_dim = (int(sensor_dim[0] / 2), int(sensor_dim[0] / 2), )
        roi_dim = (100, 50)
        cont_c2_d_avg = np.mean(pycis.get_roi(contrast_c2_d, roi_dim=roi_dim))
        phase_c2_d_avg = np.mean(pycis.get_roi(pycis.unwrap(phase_c2_d.values), roi_dim=roi_dim))
        cont_sum_d_avg = np.mean(pycis.get_roi(contrast_sum_d, roi_dim=roi_dim))
        phase_sum_d_avg = np.mean(pycis.get_roi(pycis.unwrap(phase_sum_d.values), roi_dim=roi_dim))
        cont_diff_d_avg = np.mean(pycis.get_roi(contrast_diff_d, roi_dim=roi_dim))
        phase_diff_d_avg = np.mean(pycis.get_roi(pycis.unwrap(phase_diff_d.values), roi_dim=roi_dim))

        conts_d_avg = np.array([cont_c2_d_avg,
                                cont_sum_d_avg,
                                cont_diff_d_avg])
        phase_d_avg = -pycis.wrap(np.array([phase_c2_d_avg,
                                phase_sum_d_avg,
                                phase_diff_d_avg,
                                ]))

        msize = 4
        ax_cont.plot(delay_axis_103.values, cont_analytic.values, color=col)
        ax_cont.plot(delays_103, conts_d_avg, 'd', color=col, markersize=msize, fillstyle='none')

        ax_ph.plot(delay_axis_103.values, phase_shift_analytic, color=col)
        ax_ph.plot(delays_103, phase_d_avg, 'd', color=col, markersize=msize, fillstyle='none')

        # ax.set_title(ax_title)
        # some axis formatting
        spines = ['left', 'right', 'top', 'bottom']
        for sp in spines:
            ax.spines[sp].set_color(col)
            ax.spines[sp].set_linewidth(1.5)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

    # figure formatting:
    # legend
    labels_temp = ['$0.1$ eV',
                   '$1$ eV',
                   '$5$ eV'
              ]
    labels_flow = ['$0$ kms$^{-1}$',
                   '$15$ kms$^{-1}$',
                   '$30$ kms$^{-1}$'
                   ]
    handles = [Line2D([-1, ], [-1, ], color=cols[0],),
               Line2D([-1, ], [-1, ], color=cols[1], ),
               Line2D([-1, ], [-1, ], color=cols[2], ),
               ]
    leg_temp = ax_spec.legend(handles=handles, labels=labels_temp, loc='upper left')
    leg_temp.set_title('Temperature:')
    leg_temp._legend_box.align = 'left'
    ax_spec.add_artist(leg_temp)

    leg_flow = ax_spec.legend(handles=handles, labels=labels_flow, loc='upper right')
    leg_flow.set_title('Flow:')
    leg_flow._legend_box.align = 'left'
    ax_spec.add_artist(leg_flow)

    frac_under_c, frac_over_c = -0.025, 1.1

    ax_spec.set_xlabel('$\Delta\lambda$ (nm)')
    ax_spec.set_xlim(-0.15, 0.15)
    ax_spec.set_ylabel('Spectral irradiance (arb.)')
    ax_spec.set_yticks([0])
    ax_spec.set_ylim(frac_under_c * spec_max, frac_over_c * spec_max)
    ax_spec.spines['top'].set_visible(False)
    ax_spec.spines['right'].set_visible(False)

    ax_cont.set_ylim(frac_under_c, frac_over_c)
    ax_ph.set_ylim(-0.2, np.pi)
    ax_cont.set_ylabel('Contrast')
    ax_ph.set_ylabel('Phase shift (rad)')
    ax_cont.set_yticks([0, 0.5, 1.])

    for ax in [ax_cont, ax_ph]:

        ax.vlines(delays_103, ax.get_ylim()[0], ax.get_ylim()[1], 'grey', lw=0.5, linestyles='--')
        ax.set_xlim(0, delay_axis.max() * 1e-3)
        ax.set_xlabel('Delay ($10^3$\\thinspace{}rad)')
        # add second axis for abbo thickness
        ax_2 = ax.twiny()
        ax_2.set_xlabel('$\\alpha$-BBO thickness (mm)')
        ne0, no0 = pycis.get_refractive_indices(wl0, material='a-BBO', sellmeier_coefs_source='agoptics')
        biref0 = ne0 - no0
        lwp_max = float(-delay_axis.max() * wl0 / (2 * np.pi * biref0))
        lwps = np.array([0., 5e-3, 10e-3, 15e-3])
        delays_t = 2 * np.pi * lwps * biref0 / wl0

        print(lwp_max, lwps)
        ax_2.set_xticks(lwps / lwp_max)
        ax_2.set_xticklabels(['$0$', '$5$', '$10$', '$15$'])
        ax.spines['right'].set_visible(False)
        ax_2.spines['right'].set_visible(False)

    # bottom legend
    labels_c = ['Predicted', 'Demodulated', ]
    handles_c = [Line2D([-1, ], [-1, ], color='gray'),
                 Line2D([-1, ], [-1, ], marker='d', markersize=msize, lw=0, fillstyle='none', color='gray'),
                 ]
    leg_c = ax_cont.legend(handles=handles_c, labels=labels_c, )
    ax_cont.add_artist(leg_c)

    # titles
    tits = ['(a)', '(b)', '(c)', '(d)']
    tit_pads = [None, None, 35, 35]
    for ax, tit, tit_pad in zip([ax_spec, ax1, ax_cont, ax_ph], tits, tit_pads):
        ax.set_title(tit, pad=tit_pad)
    if overwrite_output_img:
        fpath_fig = os.path.join(root, 'multi_delay_polcam_demod_test.pdf')
        fig.savefig(fpath_fig, bbox_inches='tight', transparent=True, dpi=350)
    plt.show()

    return


if __name__ == '__main__':
    plot_demod_test()