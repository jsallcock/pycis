import os
import numpy as np
import pycis.analysis.fourier_demod_2d
import xarray as xr
import matplotlib.pyplot as plt
from pycis.model import Camera, Instrument, get_spectrum_delta


inst_types = [
    'single_delay_linear',
    'single_delay_pixelated',
]

fig = plt.figure(figsize=(10, 3.5, ))
axes = fig.subplots(1, 2)

for ax, inst_type in zip(axes, inst_types):
    inst = Instrument(config=inst_type + '.yaml')
    print(inst.type)
    spectrum = get_spectrum_delta(465e-9, 5e3)
    igram = inst.capture(spectrum, )
    dim = igram.shape
    dim_show = 120
    igram_show = igram[
                 int(dim[0] / 2) - int(dim_show / 2):int(dim[0] / 2) + int(dim_show / 2),
                 int(dim[1] / 2) - int(dim_show / 2):int(dim[1] / 2) + int(dim_show / 2),
                 ]
    pycis.analysis.fourier_demod_2d.fourier_demod_2d(igram, display=False)
    igram_show.plot(x='x_pixel', y='y_pixel', vmin=0, ax=ax, cmap='gray')
    ax.set_aspect('equal')
    ax.set_title(inst_type)

plt.tight_layout()
plt.show()