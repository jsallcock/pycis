import os
import unittest
import numpy as np
import pycis
from numpy.testing import assert_almost_equal
import xarray as xr
from pycis.model import Camera, LinearPolariser, QuarterWaveplate, UniaxialCrystal, Instrument, Waveplate

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
# angle = 0

# define input spectrum
wavelength = np.linspace(460e-9, 460.05e-9, 20)
wavelength = xr.DataArray(wavelength, dims=('wavelength',), coords=(wavelength,), )
x, y = camera.get_pixel_position()

spectrum_test = xr.ones_like(wavelength * x * y, )
spectrum_test /= spectrum_test.integrate(coord='wavelength')
spectrum_test *= 1e3

# test for off-centre image subsection too
roi = {
    'x': slice(0, max(x) / 4),
    'y': slice(0, max(y) / 4),
}
spectrum_test_roi = spectrum_test.sel(roi)

spectra = [
    spectrum_test,
    spectrum_test_roi
]


class TestInstrument(unittest.TestCase):

    def test_single_delay_linear_vs_mueller(self, ):
        """
        Test that the output of the 'single_delay_linear' instrument_type is the same as for the full Mueller matrix
        calculation
        """
        camera.type = 'monochrome'

        interferometer = [
            LinearPolariser(
                orientation=0 + angle,
            ),
            UniaxialCrystal(
                thickness=5e-3,
                cut_angle=45,
                orientation=45 + angle
            ),
            LinearPolariser(
                orientation=0 + angle,
            ),
        ]

        inst = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=False)
        inst_fm = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=True)

        self.assertEqual(inst.type, 'single_delay_linear')
        self.assertEqual(inst_fm.type, 'mueller')

        for spectrum in spectra:
            igram = inst.capture(spectrum, clean=True, )
            igram_fm = inst_fm.capture(spectrum, clean=True, )
            assert_almost_equal(igram.values, igram_fm.values)

    def test_single_delay_pixelated_vs_mueller(self, ):
        """
        Test that the output of the 'single_delay_polarised' instrument_type is the same as for the full Mueller matrix
        calculation
        """
        camera.type = 'monochrome_polarised'
        interferometer = [
            LinearPolariser(
                orientation=0 + angle
            ),
            UniaxialCrystal(
                thickness=5e-3,
                cut_angle=0,
                orientation=45 + angle,
            ),
            QuarterWaveplate(
                orientation=90 + angle,
            )
        ]
        kwargs = {
            'camera': camera,
            'optics': optics,
            'interferometer': interferometer,
        }
        inst = Instrument(**kwargs, force_mueller=False)
        inst_fm = Instrument(**kwargs, force_mueller=True)
        self.assertEqual(inst.type, 'single_delay_pixelated')
        self.assertEqual(inst_fm.type, 'mueller')

        for spectrum in spectra:
            igram = inst.capture(spectrum, clean=True, )
            igram_fm = inst_fm.capture(spectrum, clean=True, )
            assert_almost_equal(igram.values, igram_fm.values)

    def test_double_delay_linear_vs_mueller(self, ):
        """
        Test that the output of the 'double_delay_linear' instrument_type is the same as for the full Mueller matrix
        calculation
        """
        camera.type = 'monochrome'
        interferometer = [
            LinearPolariser(
                orientation=45 + angle,
            ),
            UniaxialCrystal(
                orientation=0 + angle,
                thickness=8.e-3,
                cut_angle=45,
            ),
            UniaxialCrystal(
                orientation=45 + angle,
                thickness=9.8e-3,
                cut_angle=45,
            ),
            LinearPolariser(
                orientation=0 + angle,
            ),
        ]
        instrument = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=False)
        instrument_fm = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=True)
        self.assertEqual(instrument.type, 'double_delay_linear')
        self.assertEqual(instrument_fm.type, 'mueller')

        for spectrum in spectra:
            igram = instrument.capture(spectrum, clean=True, )
            igram_fm = instrument_fm.capture(spectrum, clean=True, )
            assert_almost_equal(igram.values, igram_fm.values)

    def test_triple_delay_linear_vs_mueller(self, ):
        """
        Test that the output of the 'triple_delay_linear' instrument_type is the same as for the full Mueller matrix
        calculation
        """
        camera.type = 'monochrome'
        interferometer = [
            LinearPolariser(
                orientation=22.5 + angle,
            ),
            UniaxialCrystal(
                orientation=0 + angle,
                thickness=8.e-3,
                cut_angle=45,
            ),
            UniaxialCrystal(
                orientation=45 + angle,
                thickness=9.8e-3,
                cut_angle=45,
            ),
            LinearPolariser(
                orientation=0 + angle,
            ),
        ]
        instrument = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=False)
        instrument_fm = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=True)
        self.assertEqual(instrument.type, 'triple_delay_linear')
        self.assertEqual(instrument_fm.type, 'mueller')

        for spectrum in spectra:
            igram = instrument.capture(spectrum, clean=True, )
            igram_fm = instrument_fm.capture(spectrum, clean=True, )
            assert_almost_equal(igram.values, igram_fm.values)

    def test_quad_delay_linear_vs_mueller(self, ):
        """
        Test that the output of the 'quad_delay_linear' instrument_type is the same as for the full Mueller matrix
        calculation
        """
        camera.type = 'monochrome'
        interferometer = [
            LinearPolariser(
                orientation=22.5 + angle,
            ),
            UniaxialCrystal(
                orientation=0 + angle,
                thickness=8.e-3,
                cut_angle=45,
            ),
            UniaxialCrystal(
                orientation=45 + angle,
                thickness=9.8e-3,
                cut_angle=45,
            ),
            LinearPolariser(
                orientation=22.5 + angle,
            ),
        ]
        instrument = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=False)
        instrument_fm = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=True)
        self.assertEqual(instrument.type, 'quad_delay_linear')
        self.assertEqual(instrument_fm.type, 'mueller')

        for spectrum in spectra:
            igram = instrument.capture(spectrum, clean=True, )
            igram_fm = instrument_fm.capture(spectrum, clean=True, )
            assert_almost_equal(igram.values, igram_fm.values)

    def test_double_delay_pixelated_vs_mueller(self, ):
        """
        Test that the output of the 'double_delay_pixelated' instrument_type is the same as for the full Mueller matrix
        calculation
        """
        camera.type = 'monochrome_polarised'
        interferometer = [
            LinearPolariser(
                orientation=45 + angle,
            ),
            UniaxialCrystal(
                orientation=0 + angle,
                thickness=8.e-3,
                cut_angle=45,
            ),
            UniaxialCrystal(
                orientation=45 + angle,
                thickness=9.8e-3,
                cut_angle=45,
            ),
            QuarterWaveplate(
                orientation=90 + angle,
            ),
        ]
        instrument = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=False)
        instrument_fm = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=True)
        self.assertEqual(instrument.type, 'double_delay_pixelated')
        self.assertEqual(instrument_fm.type, 'mueller')

        for spectrum in spectra:
            igram = instrument.capture(spectrum, clean=True, )
            igram_fm = instrument_fm.capture(spectrum, clean=True, )
            assert_almost_equal(igram.values, igram_fm.values)

    def test_triple_delay_pixelated_vs_mueller(self, ):
        """
        Test that the output of the 'triple_delay_pixelated' instrument_type is the same as for the full Mueller matrix
        calculation
        """
        camera.type = 'monochrome_polarised'
        interferometer = [
            LinearPolariser(
                orientation=22.5 + angle,
            ),
            UniaxialCrystal(
                orientation=0 + angle,
                thickness=8.e-3,
                cut_angle=45,
            ),
            UniaxialCrystal(
                orientation=45 + angle,
                thickness=9.8e-3,
                cut_angle=45,
            ),
            QuarterWaveplate(
                orientation=90 + angle,
            ),
        ]
        instrument = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=False)
        instrument_fm = Instrument(camera=camera, optics=optics, interferometer=interferometer, force_mueller=True)
        self.assertEqual(instrument.type, 'triple_delay_pixelated')
        self.assertEqual(instrument_fm.type, 'mueller')

        for spectrum in spectra:
            igram = instrument.capture(spectrum, clean=True, )
            igram_fm = instrument_fm.capture(spectrum, clean=True, )
            assert_almost_equal(igram.values, igram_fm.values)

    def test_read_config_write_config(self, ):
        inst_1 = pycis.Instrument('single_delay_pixelated.yaml')
        testpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test.yaml')
        inst_1.write_config(testpath)
        inst_2 = pycis.Instrument(testpath)
        os.remove(testpath)
        assert inst_1 == inst_2

    # import matplotlib.pyplot as plt
    # plt.figure()
    # igram.plot(x='x', y='y')
    # plt.figure()
    # igram_fm.plot(x='x', y='y')
    # plt.figure()
    # (igram.astype(float) - igram_fm.astype(float)).plot(x='x', y='y')
    # plt.show()


if __name__ == '__main__':
    unittest.main()
