import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pycis import Camera, LinearPolariser, UniaxialCrystal, Instrument, get_spectrum_delta


inst_types = [
    'single_delay_linear',
    'single_delay_pixelated',
]

fig = plt.figure(figsize=(10, 3.5, ))
axes = fig.subplots(1, 2)

for ax, inst_type in zip(axes, inst_types):

    inst = Instrument('demo_config_' + inst_type + '.yaml')
    spectrum = get_spectrum_delta(465e-9, 5e3)
    igram = inst.capture(spectrum, )

    igram.plot(x='x_pixel', y='y_pixel', vmin=0, vmax=1.2 * float(igram.max()), ax=ax, cmap='gray')
    ax.set_aspect('equal')
    ax.set_title(inst_type)

plt.tight_layout()
plt.show()