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
camera = Camera(bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise, mode='mono')

# define instrument optics
optics = [17e-3, 105e-3, 150e-3, ]

# random rotation angle to be added to all interferometer components
angle = np.random.rand() * 180 * np.pi / 180

# define input spectrum
wavelength = np.linspace(460e-9, 460.05e-9, 20)
wavelength = xr.DataArray(wavelength, dims=('wavelength',), coords=(wavelength,), )
x, y = camera.calculate_pixel_position()

spectrum = xr.ones_like(x * y * wavelength, )
spectrum /= spectrum.integrate(dim='wavelength')
spectrum *= 5e3


class TestInstrument(unittest.TestCase):
    def test_single_delay_linear(self, ):
        """
        Test that the output of the 'single_delay_linear' instrument_type is the same as for the full Mueller matrix
        calculation

        """
        camera.mode = 'mono'

        interferometer = [LinearPolariser(0 + angle, ),
                          UniaxialCrystal(np.pi / 4 + angle, 5e-3, np.pi / 4, contrast=0.5, ),
                          LinearPolariser(0 + angle, ),
                          ]

        inst = Instrument(camera, optics, interferometer, force_mueller=False)
        inst_fm = Instrument(camera, optics, interferometer, force_mueller=True)

        self.assertEqual(inst.instrument_type, 'single_delay_linear')
        self.assertEqual(inst_fm.instrument_type, 'mueller')

        igram = inst.capture(spectrum, clean=True, )
        igram_fm = inst_fm.capture(spectrum, clean=True, )

        assert_almost_equal(igram.values, igram_fm.values)

    def test_single_delay_polarised(self, ):
        """
        Test that the output of the 'single_delay_polarised' instrument_type is the same as for the full Mueller matrix
        calculation

        """
        camera.mode = 'mono_polarised'

        interferometer = [LinearPolariser(0 + angle, ),
                          UniaxialCrystal(np.pi / 4 + angle, 5e-3, 0, contrast=0.5, ),
                          QuarterWaveplate(np.pi / 2 + angle, )
                          ]

        inst = Instrument(camera, optics, interferometer, force_mueller=False)
        inst_fm = Instrument(camera, optics, interferometer, force_mueller=True)

        self.assertEqual(inst.instrument_type, 'single_delay_polarised')
        self.assertEqual(inst_fm.instrument_type, 'mueller')

        igram = inst.capture(spectrum, clean=True, )
        igram_fm = inst_fm.capture(spectrum, clean=True, )

        assert_almost_equal(igram.values, igram_fm.values)

    def test_multi_delay_polarised(self, ):
        """
        Test that the output of the 'multi_delay_polarised' instrument_type is the same as for the full Mueller matrix
        calculation

        """
        camera.mode = 'mono_polarised'

        interferometer = [LinearPolariser(0 + angle, ),
                          UniaxialCrystal(np.pi / 4 + angle, 5e-3, np.pi / 4, contrast=0.5, ),
                          LinearPolariser(0 + angle, ),
                          UniaxialCrystal(np.pi / 4 + angle, 5e-3, 0, contrast=0.5, ),
                          QuarterWaveplate(np.pi / 2 + angle, )
                          ]

        instrument = Instrument(camera, optics, interferometer, force_mueller=False)
        instrument_fm = Instrument(camera, optics, interferometer, force_mueller=True)

        self.assertEqual(instrument.instrument_type, 'multi_delay_polarised')
        self.assertEqual(instrument_fm.instrument_type, 'mueller')

        igram = instrument.capture(spectrum, clean=True, )
        igram_fm = instrument_fm.capture(spectrum, clean=True, )

        assert_almost_equal(igram.values, igram_fm.values)


if __name__ == '__main__':
    unittest.main()
