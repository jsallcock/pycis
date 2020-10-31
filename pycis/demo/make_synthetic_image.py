import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pycis import Camera, LinearPolariser, UniaxialCrystal, Instrument

"""
make a simple 'classic' coherence imaging instrument with a single delay, observing a uniform, unpolarised spectral 
scene.
"""

# define camera
bit_depth = 12
sensor_format = (500, 500, )
pixel_size = 6.5e-6 * 2
qe = 0.35
epercount = 0.46  # [e / count]
cam_noise = 2.5
camera = Camera(bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise, mode='mono')

# define instrument optics -- just focal lengths right now
optics = [17e-3, 105e-3, 150e-3, ]

# define the interferometer
angle = 35.2 * np.pi / 180  # an arbitrary rotation angle to be added to all components
polariser = LinearPolariser(0 + angle, )

thickness = 8e-3
cut_angle = np.pi / 4
contrast = 0.5  # manually set the instrument contrast contribution of this crystal
displacer_plate = UniaxialCrystal(np.pi / 4 + angle, thickness, cut_angle, contrast=contrast, )
interferometer = [polariser, displacer_plate, polariser, ]

instrument = Instrument(camera, optics, interferometer, force_mueller=False)
print(instrument.instrument_type)

# specify input spectrum -- xr.DataArray with dimensions 'x' and 'y' corresponding to position on the camera's sensor
# plane, with units in m. And also with dimension 'wavelength' in units m. Spectrum units are ph / m.
wavelength = np.linspace(460e-9, 460.05e-9, 5)
wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), )
x, y = camera.get_pixel_position()  # camera method returns the camera's pixel x and y positions as DataArrays

spectrum = xr.ones_like(x * y * wavelength, )
spectrum /= spectrum.integrate(dim='wavelength')
spectrum *= 5e3

igram = instrument.capture(spectrum, )
igram.plot(x='x_pixel', y='y_pixel', vmin=0, )
plt.show()
