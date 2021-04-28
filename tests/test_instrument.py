import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import xarray as xr
from pycis.model import Camera, LinearPolariser, QuarterWaveplate, UniaxialCrystal, Instrument

# define camera
bit_depth = 12
sensor_format = (100, 100,)
pixel_size = 6.5e-6 * 2
qe = 0.35
epercount = 0.46  # [e / count]
cam_noise = 2.5
camera = Camera(sensor_format, pixel_size, bit_depth, qe, epercount, cam_noise, type='monochrome')

# define instrument optics
optics = [17e-3, 105e-3, 150e-3, ]

# random rotation angle to be added to all interferometer components
angle = np.random.rand() * 180

# define input spectrum
wavelength = np.linspace(460e-9, 460.05e-9, 20)
wavelength = xr.DataArray(wavelength, dims=('wavelength',), coords=(wavelength,), )
x, y = camera.get_pixel_position()

spectrum = xr.ones_like(wavelength, )
spectrum /= spectrum.integrate(coord='wavelength')
spectrum *= 1e3


class TestInstrument(unittest.TestCase):
    def test_single_delay_linear(self, ):
        """
        Test that the output of the 'single_delay_linear' instrument_type is the same as for the full Mueller matrix
        calculation

        """
        camera.type = 'monochrome'

        interferometer = [LinearPolariser(0 + angle, ),
                          UniaxialCrystal(45 + angle, 5e-3, 45, contrast=0.5, ),
                          LinearPolariser(0 + angle, ),
                          ]

        inst = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=False)
        inst_fm = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=True)

        self.assertEqual(inst.type, 'single_delay_linear')
        self.assertEqual(inst_fm.type, 'mueller')

        igram = inst.capture(spectrum, clean=True, )
        igram_fm = inst_fm.capture(spectrum, clean=True, )

        assert_almost_equal(igram.values, igram_fm.values)

    def test_single_delay_pixelated(self, ):
        """
        Test that the output of the 'single_delay_polarised' instrument_type is the same as for the full Mueller matrix
        calculation

        """
        camera.type = 'monochrome_polarised'

        interferometer = [LinearPolariser(0 + angle),
                          UniaxialCrystal(45 + angle, 5e-3, 0, contrast=0.5, ),
                          QuarterWaveplate(90 + angle, )
                          ]

        inst = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=False)
        inst_fm = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=True)

        self.assertEqual(inst.type, 'single_delay_pixelated')
        self.assertEqual(inst_fm.type, 'mueller')

        igram = inst.capture(spectrum, clean=True, )
        igram_fm = inst_fm.capture(spectrum, clean=True, )

        assert_almost_equal(igram.values, igram_fm.values)

    def test_multi_delay_pixelated(self, ):
        """
        Test that the output of the 'multi_delay_polarised' instrument_type is the same as for the full Mueller matrix
        calculation

        """
        camera.type = 'monochrome_polarised'

        interferometer = [LinearPolariser(0 + angle, ),
                          UniaxialCrystal(45 + angle, 5e-3, np.pi / 4, contrast=0.5, ),
                          LinearPolariser(0 + angle, ),
                          UniaxialCrystal(45 + angle, 5e-3, 0, contrast=0.5, ),
                          QuarterWaveplate(90 + angle, )
                          ]

        instrument = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=False)
        instrument_fm = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=True)

        self.assertEqual(instrument.type, 'multi_delay_pixelated')
        self.assertEqual(instrument_fm.type, 'mueller')

        igram = instrument.capture(spectrum, clean=True, )
        igram_fm = instrument_fm.capture(spectrum, clean=True, )

        assert_almost_equal(igram.values, igram_fm.values)


if __name__ == '__main__':
    unittest.main()
