import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pycis import Camera, LinearPolariser, UniaxialCrystal, QuarterWaveplate, Instrument
import time
import os

bit_depth = 12
sensor_format = (50, 50, )
pixel_size = 6.5e-6 * 20
qe = 0.35
epercount = 0.46  # [e / count]
cam_noise = 2.5
camera = Camera(bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise, polarised=False)
camera_pol = Camera(bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise, polarised=True)

optics = [17e-3, 105e-3, 150e-3, ]

pol = LinearPolariser(0, )
contrast = 0.5
wp = UniaxialCrystal(np.pi / 4, 5e-3, 0, contrast=contrast, )
qwp = QuarterWaveplate(np.pi / 2, )
interferometer_1 = [pol, wp, qwp, ]
instrument_1 = Instrument(camera_pol, optics, interferometer_1)
instrument_2 = Instrument(camera_pol, optics, interferometer_1, force_mueller=True)
print(instrument_1.instrument_type)
print(instrument_2.instrument_type)

wavelength = np.linspace(460e-9, 460.05e-9, 5)
wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), )
x, y = camera.calculate_pixel_position()

spec = xr.ones_like(x * y * wavelength, )
spec /= spec.integrate(dim='wavelength')
spec *= 5e3
# spec = spec.chunk({'x': 100, 'y': 100, })

s = time.time()
igram_1 = instrument_1.capture_image(spec, )
igram_2 = instrument_2.capture_image(spec, )
igram_1 = igram_1.load()
igram_2 = igram_2.load()
e = time.time()
print(e - s, 'sec')

