import yaml
import numpy as np
import matplotlib.pyplot as plt
from pycis import Instrument, get_spectrum_delta, fft2_im

inst = Instrument(config='/Users/jsallcock/py_repo/pycis/pycis/vis/pycis_config_MWI_CI.yaml')
igram = inst.capture(get_spectrum_delta(465e-9, 5e3), )
psd = np.log10(np.abs(fft2_im(igram)) ** 2)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
igram.plot(x='x', y='y', ax=ax1)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
psd.plot(x='freq_x', y='freq_y', ax=ax2)

d = 3.45e-6
contrast = np.abs(np.sin(np.pi * psd.freq_x * d) / (np.pi * psd.freq_x * d) * np.sin(np.pi * psd.freq_y * d) / (np.pi * psd.freq_y * d))

CS = ax2.contour(contrast.freq_x.values, contrast.freq_y.values, contrast.values.T, colors='w',
                             levels=[0.5, 0.6, 0.7, 0.8, 0.9, ], linewidths=0.75, )
ax2.clabel(CS, CS.levels, fmt=None, fontsize=8)

for ax in [ax1, ax2]:
    ax.set_aspect('equal')
    for sp in ax.spines:
        ax.spines[sp].set_visible(False)

plt.show()
