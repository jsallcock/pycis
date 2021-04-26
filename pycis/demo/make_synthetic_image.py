import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pycis import Camera, LinearPolariser, UniaxialCrystal, Instrument

# load instrument via config file
inst = Instrument('demo_config_single_delay_pixelated.yaml')

# specify input spectrum -- xr.DataArray with dimensions 'x' and 'y' corresponding to position on the camera's sensor
# plane, with units in m. And also with dimension 'wavelength' in units m. Spectrum units are ph / m.
wavelength = np.linspace(460e-9, 460.05e-9, 5)
wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), )

spectrum = xr.ones_like(inst.camera.x * inst.camera.y * wavelength, )
spectrum /= spectrum.integrate(coord='wavelength')
spectrum *= 5e3

igram = inst.capture(spectrum, )

igram.plot(x='x_pixel', y='y_pixel', vmin=0, )
plt.show()
