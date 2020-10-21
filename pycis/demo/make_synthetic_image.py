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
l = 5e-3

pol_sp = LinearPolariser(0, )
sp = SavartPlate(np.pi / 4, l, )

pol_dp = LinearPolariser(np.pi / 4)
dp = UniaxialCrystal(np.pi / 2, l, np.pi / 4, )

interferometer_dp = [pol_dp, dp, pol_dp, ]
interferometer_sp = [pol_sp, sp, pol_sp, ]
instrument_dp = Instrument(camera, optics, interferometer_dp, )
instrument_sp = Instrument(camera, optics, interferometer_sp, )
print(instrument_dp.instrument_type)
print(instrument_sp.instrument_type)

wavelength = np.linspace(460e-9, 460.05e-9, 5)
wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), )
x, y = camera.calculate_pixel_position()

spec = xr.ones_like(x * y * wavelength, )
spec /= spec.integrate(dim='wavelength')
spec *= 5e3
# spec = spec.chunk({'x': 100, 'y': 100, })

s = time.time()
igram_dp = instrument_dp.capture_image(spec, )
igram_sp = instrument_sp.capture_image(spec, )
igram_dp = igram_dp.load()
igram_sp = igram_sp.load()
e = time.time()
print(e - s, 'sec')

