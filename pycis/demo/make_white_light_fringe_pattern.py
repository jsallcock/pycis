import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pycis import Camera, LinearPolariser, UniaxialCrystal, Instrument


"""
This has no use for plasma diagnostics, it just looks nice. 

the RGB camera mode hasn't been tested, don't use this for anything that matters!
"""


# define camera
bit_depth = 12
sensor_format = (600, 200, )
pixel_size = 6.5e-6
qe = 0.35
epercount = 0.46  # [e / count]
cam_noise = 2.5
camera = Camera(bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise, mode='rgb')

# define instrument optics -- just focal lengths right now
optics = [17e-3, 105e-3, 15e-3, ]

# define the interferometer
angle = 35.2 * np.pi / 180  # an arbitrary rotation angle to be added to all components

thickness = 7.5e-3
cut_angle = 0
contrast = 0.5  # manually set the instrument contrast contribution of this crystal
uc = UniaxialCrystal(np.pi / 4 + angle, thickness, 90 * np.pi / 180, )
interferometer = [LinearPolariser(0 + angle, ),
                  uc,
                  LinearPolariser(np.pi / 2 + angle, ), ]

instrument = Instrument(camera, optics, interferometer, force_mueller=False)

wavelength = np.linspace(100e-9, 1000e-9, 200)
wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), )
x, y = camera.get_pixel_position()  # camera method returns the camera's pixel x and y positions as DataArrays

spectrum = xr.ones_like(x * y * wavelength, )
spectrum /= spectrum.integrate(dim='wavelength')
spectrum *= 5e3
spectrum = spectrum.chunk({'x': 50, 'y': 50})

igram = instrument.capture(spectrum, clean=True, )

fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow(igram.values.T)
plt.savefig('white_light_fringes.png')
plt.close()
