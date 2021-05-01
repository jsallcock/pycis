import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pycis import Camera, LinearPolariser, UniaxialCrystal, SavartPlate, Instrument

"""
This has no use but it looks nice. 

the RGB camera type hasn't been tested, don't use this for anything that matters!
"""

# define camera
bit_depth = 12
sensor_format = (600, 200, )
pixel_size = 6.5e-6
qe = 0.35
epercount = 0.46  # [e / count]
cam_noise = 2.5
camera = Camera(sensor_format, pixel_size, bit_depth, qe, epercount, cam_noise, type='rgb')

# define instrument optics -- just focal lengths right now
optics = [17e-3, 105e-3, 25e-3, ]

interferometer = [
    LinearPolariser(0, ),
    UniaxialCrystal(45, 20e-3, 90, ),
    LinearPolariser(0, ),
]

inst = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=False)
print(inst.type)

wavelength = np.linspace(400e-9, 750e-9, 400)
wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), )

spectrum = xr.ones_like(inst.camera.x * inst.camera.y * wavelength, )
spectrum /= spectrum.integrate(coord='wavelength')
spectrum *= 5e3
spectrum = spectrum.chunk({'x': 50, 'y': 50})

igram = inst.capture(spectrum, clean=True, )

fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow(igram.values.T, vmin=0, )
plt.savefig('white_light_fringes_aligned_polarisers.png', bbox_inches='tight', pad_inches=0)
plt.close()
