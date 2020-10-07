import numpy as np
import xarray as xr
import pycis
import os

# define coherence imaging instrument
bit_depth = 12
sensor_format = (1250, 1000)
pixel_size = 6.5e-6 * 2
qe = 0.35
epercount = 0.46  # [e / count]
cam_noise = 2.5
camera = pycis.Camera(bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise)

optics = [17e-3, 105e-3, 150e-3, ]

pol_1 = pycis.LinearPolariser(0)
wp_1 = pycis.UniaxialCrystal(np.pi / 4, 10e-3, 0, )
pol_2 = pycis.LinearPolariser(0)
interferometer = [pol_1, wp_1, pol_2]
inst = pycis.Instrument(camera, optics, interferometer)

wavelength = np.linspace(460e-9, 460.05e-9, 3)
wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), )
x, y = inst.calculate_pixel_position()

spec = xr.ones_like(x * y * wavelength, )
spec /= spec.integrate(dim='wavelength')
spec *= 5e3

print(os.path.dirname(os.path.realpath(__file__)))

# fpath_spec = os.path.join(, 'spec.nc')

spec.to_netcdf(fpath_spec)
spec = xr.open_dataarray(fpath_spec, chunks={'x': 200, 'y': 200, })

igram = inst.capture_image(spec, )
igram.load()

