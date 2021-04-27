import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pycis import Camera, LinearPolariser, UniaxialCrystal, Instrument


inst_types = [
    'single_delay_linear',
    'single_delay_pixelated',
]

fig = plt.figure(figsize=(10, 3.5, ))
axes = fig.subplots(1, 2)

for ax, inst_type in zip(axes, inst_types):
    print(inst_type)
    inst = Instrument('demo_config_' + inst_type + '.yaml')
    print(inst.type)

    # specify input spectrum -- xr.DataArray with dimensions 'x' and 'y' corresponding to position on the camera's sensor
    # plane, with units in m. And also with dimension 'wavelength' in units m. Spectrum units are ph / m.
    wavelength = np.linspace(460e-9, 460.05e-9, 3)
    wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), )
    spectrum = xr.ones_like(inst.camera.x * inst.camera.y * wavelength, )
    spectrum /= spectrum.integrate(coord='wavelength')
    spectrum *= 5e3

    igram = inst.capture(spectrum, )
    igram.plot(x='x_pixel', y='y_pixel', vmin=0, vmax=1.2 * float(igram.max()), ax=ax, cmap='gray')
    ax.set_aspect('equal')
    ax.set_title(inst_type)

plt.tight_layout()
plt.show()