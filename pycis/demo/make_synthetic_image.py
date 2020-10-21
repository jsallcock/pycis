import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pycis import Camera, LinearPolariser, UniaxialCrystal, QuarterWaveplate, Instrument, SavartPlate
import time
import os

"""
currently using this script for quick tests
"""

bit_depth = 12
sensor_format = (500, 500, )
pixel_size = 6.5e-6 * 2
qe = 0.35
epercount = 0.46  # [e / count]
cam_noise = 2.5
camera = Camera(bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise, polarised=False)
camera_pol = Camera(bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise, polarised=True)

optics = [17e-3, 105e-3, 150e-3, ]

or_int = 35.2323534663 * np.pi / 180
pol = LinearPolariser(0 + or_int, )
wp = UniaxialCrystal(np.pi / 4 + or_int, 5e-3, 0, )
dp = UniaxialCrystal(np.pi / 4 + or_int, 5e-3, np.pi / 4)
qwp = QuarterWaveplate(np.pi / 2 + or_int, )

interferometer = [pol, dp, pol, wp, qwp, ]

instrument = Instrument(camera_pol, optics, interferometer, )
print(instrument.instrument_type)

wavelength = np.linspace(460e-9, 460.05e-9, 5)
wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), )
x, y = camera.calculate_pixel_position()

spec = xr.ones_like(x * y * wavelength, )
spec /= spec.integrate(dim='wavelength')
spec *= 5e3
# spec = spec.chunk({'x': 100, 'y': 100, })

s = time.time()
igram = instrument.capture_image(spec, )
igram = igram.load()
e = time.time()
print(e - s, 'sec')

