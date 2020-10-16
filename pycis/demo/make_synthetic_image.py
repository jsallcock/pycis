import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pycis import Camera, LinearPolariser, UniaxialCrystal, QuarterWaveplate, Instrument
import time
import os

bit_depth = 12
sensor_format = (500, 500)
pixel_size = 6.5e-6 * 2
qe = 0.35
epercount = 0.46  # [e / count]
cam_noise = 2.5
camera = Camera(bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise, polarised=True)

optics = [17e-3, 105e-3, 150e-3, ]

interferometer_orientation = 0
pol_1 = LinearPolariser(0 + interferometer_orientation)
wp_1 = UniaxialCrystal(np.pi / 4 + interferometer_orientation, 10e-3, 0, )
qwp = QuarterWaveplate(np.pi / 2 + interferometer_orientation, )
pol_2 = LinearPolariser(0 + interferometer_orientation)
interferometer = [pol_1, wp_1, pol_1, wp_1, qwp, ]
inst = Instrument(camera, optics, interferometer)
print(inst.instrument_type)
#
# wavelength = np.linspace(460e-9, 460.05e-9, 3)
# wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), )
# x, y = camera.calculate_pixel_position()
#
# spec = xr.ones_like(x * y * wavelength, )
# spec /= spec.integrate(dim='wavelength')
# spec *= 5e3
# spec = spec.chunk({'x': 100, 'y': 100, })
#
# s = time.time()
# igram = inst.capture_image(spec, )
# igram.load()
# e = time.time()
# print(e - s, 'sec')

