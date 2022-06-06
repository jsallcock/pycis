import os
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import xarray as xr
from pycis.model import get_pixelated_phase_mask, Instrument, get_spectrum_delta
from pycis.analysis import wrap, demod_triple_delay_pixelated


class TestDemodPixelated(unittest.TestCase):

    def test_demod_triple_delay_pixelated(self):
        """
        Test that the output of the 'demod_triple_delay_pixelated' close to expected result

        Systematic differences will always exist between the predicted and Fourier-demodulated profiles. e.g. due to
        Fourier domain windowing and quantization error. Comparing only a central region-of-interest (ROI) mitigates the
        effects of the Fourier artefacts while relaxing the precision tolerance skirts the quantization errors.

        Test the coherence rather than the phase to avoid 2-pi errors from np.assert_almost_equal.
        """

        inst = Instrument(config='triple_delay_pixelated.yaml')
        wl0 = 465e-9
        spectrum = get_spectrum_delta(wl0, 5e3)
        igram = inst.capture(spectrum, clean=True)

        fringe_freq = inst.retarders[0].get_fringe_frequency(wl0, inst.optics[-1])
        delay = inst.get_delay(wl0, igram.x, igram.y)

        dc, phase, contrast = demod_triple_delay_pixelated(igram, fringe_freq)

        # perform test over central region of interest (ROI)
        xmax = max(igram.x)
        ymax = max(igram.y)
        roi = {
            'x': slice(-xmax / 4, xmax / 4),
            'y': slice(-ymax / 4, ymax / 4),
        }
        ones_roi = xr.ones_like(igram).sel(roi)

        decimal = 2  # chosen empirically based on typical quantization errors
        for ii_delay in range(3):
            phase_roi = phase[ii_delay].sel(roi)
            contrast_roi = contrast[ii_delay].sel(roi)
            delay_roi = delay[ii_delay].sel(roi)

            coherence_demod = (contrast_roi * np.exp(1j * phase_roi)).values
            coherence_predicted = np.exp(1j * delay_roi).values

            err_msg_contrast = 'Demod failed: contrast error at delay' + str(ii_delay + 1) + '/3'
            assert_almost_equal(contrast_roi.values, ones_roi, decimal=decimal, err_msg=err_msg_contrast)

            err_msg_coherence = 'Demod failed: coherence error at delay' + str(ii_delay + 1) + '/3'
            assert_almost_equal(coherence_demod, coherence_predicted, decimal=decimal, err_msg=err_msg_coherence)


if __name__ == '__main__':
    unittest.main()
