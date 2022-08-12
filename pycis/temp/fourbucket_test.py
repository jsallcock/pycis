import os
import numpy as np
# import pycis.analysis.fourier_demod_2d
import xarray as xr
import matplotlib.pyplot as plt
from pycis.model import Camera, Instrument, get_spectrum_delta
from pycis.analysis import demod_single_delay_pixelated, demod_single_delay_pixelated_mod

fig = plt.figure(figsize=(10, 3.5, ))
axes = fig.subplots(1, 2)

inst = Instrument(config='single_delay_pixelated' + '.yaml')
inst.retarders[0].cut_angle = 45
inst.retarders[0].thickness = 8e-3
inst.retarders[0].contrast_inst = 0.75
inst.optics[-2] = 150e-3

spectrum = get_spectrum_delta(465e-9, 5e3)
igram = inst.capture(spectrum, )

dc, phase, contrast = demod_single_delay_pixelated(igram)
dc_mod, phase_mod, contrast_mod = demod_single_delay_pixelated_mod(igram)

contrast.plot(x='x', y='y', vmin=0, vmax=1, ax=axes[0])
contrast_mod.plot(x='x', y='y', vmin=0, vmax=1, ax=axes[1])

# igram.plot(x='x_pixel', y='y_pixel', vmin=0, ax=ax, cmap='gray')
for ax in axes:
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()