import xarray as xr
import numpy as np

s = np.array([1, 0, 0, 0])
s = xr.DataArray(s, dims=('stokes', ), )
s.plot(x='wavelength', hue='stokes')


