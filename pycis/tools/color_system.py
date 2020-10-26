import numpy as np
import xarray as xr
from os.path import join, abspath, dirname

"""
rough conversion between intensity spectrum of light and perceived colour in RGB format.
    
    
code mostly taken from:
https://scipython.com/blog/converting-a-spectrum-to-a-colour/

with minor changes for wavelength units and xarray compatibility.
"""


def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))


class ColourSystem:
    """A class representing a colour system.

    A colour system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".

    """

    # The CIE colour matching function for 380 - 780 nm in 5 nm intervals
    wavelength = np.arange(380, 781, 5) * 1e-9
    wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), attrs={'units': 'm'}, )

    fpath = join(dirname(abspath(__file__)), 'cie-cmf.txt')
    cmf = xr.DataArray(np.loadtxt(fpath, usecols=(1, 2, 3)), dims=('wavelength', 'cmf'),
                       coords=(wavelength, np.arange(3), ))

    def __init__(self, red, green, blue, white):
        """Initialise the ColourSystem object.

        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the colour system.

        """

        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """Transform from xyz to rgb representation of colour.

        The output rgb components are normalized on their maximum
        value. If xyz is out the rgb gamut, it is desaturated until it
        comes into gamut.

        By default, fractional rgb components are returned; if
        out_fmt='html', the HTML hex string '#rrggbb' is returned.

        """

        # rgb = self.T.dot(xyz)

        t_xr = xr.DataArray(self.T, dims=('rgb', 'cmf'), coords=(np.arange(3), np.arange(3), ))
        rgb = xr.dot(t_xr, xyz)

        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = - np.min(rgb)
            rgb += w
        if not np.all(rgb == 0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)

        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """Convert from fractional rgb values to HTML-style hex string."""

        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def spec_to_xyz(self, spec):
        """Convert a spectrum to an xyz point.

        The spectrum must be on the same grid of points as the colour-matching
        function, self.cmf: 380-780 nm in 5 nm steps.

        """
        cmf_i = self.cmf.interp_like(spec)
        XYZ = (spec * cmf_i).sum(dim='wavelength')
        den = XYZ.sum()
        if den == 0.:
            return XYZ
        return XYZ / den

    def spec_to_rgb(self, spec, out_fmt=None):
        """Convert a spectrum to an rgb value."""

        xyz = self.spec_to_xyz(spec)
        return self.xyz_to_rgb(xyz, out_fmt)


illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
cs_hdtv = ColourSystem(red=xyz_from_xy(0.67, 0.33),
                       green=xyz_from_xy(0.21, 0.71),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)

cs_smpte = ColourSystem(red=xyz_from_xy(0.63, 0.34),
                        green=xyz_from_xy(0.31, 0.595),
                        blue=xyz_from_xy(0.155, 0.070),
                        white=illuminant_D65)

cs_srgb = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)


# if __name__ == '__main__':
#     # wl = np.linspace(300, 700, 1000)
#     # wavelength = np.arange(380., 781., 5)
#     wavelength = np.linspace(380e-9, 781e-9, 1000)
#     wavelength = xr.DataArray(wavelength, coords=(wavelength, ), dims=('wavelength', ), )
#
#     x = np.linspace(0, 1, 100)
#     x = xr.DataArray(x, coords=(x, ), dims=('x', ), )
#
#     y = np.linspace(0, 1, 100)
#     y = xr.DataArray(y, coords=(y,), dims=('y',), )
#
#     wl_0 = xr.DataArray(np.linspace(400e-9, 700e-9, 100), coords=(x, ), dims=('x', ), )
#     sigma = xr.DataArray(np.linspace(0.1e-9, 100e-9, 100), coords=(y, ), dims=('y', ), )
#     spectrum = 10 * np.exp(-1 / 2 * ((wavelength - wl_0) / sigma) ** 2)
#     spectrum /= spectrum.integrate(dim='wavelength')
#     # spectrum = xr.DataArray(spectrum, coords=(wavelength, x, ), dims=('wavelength', 'x', ), )
#
#     cs = cs_smpte
#     rgb = cs.spec_to_rgb(spectrum, )

