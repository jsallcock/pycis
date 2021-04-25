Model
=====
A framework for modelling CI images built using the `xarray library <http://xarray.pydata.org/en/stable/>`_ (xr) for
multi-dimensional arrays with labelled dimensions.

The Mueller matrix model makes it simple to model arbitrary interferometer configurations observing
scenes with arbitrary spectral and polarisation properties.

For some interferometer layouts there are simple analytical expressions for the interferogram. These analytical
shortcuts are configured as instrument types in pycis. This functionality makes for faster evaluation and also allows
for methods that return important instrument parameters like interferometer delay(s) across the image and fringe
frequency.

Conventions
-----------
+ Mueller matrices are xr.DataArray instances with dimensions that include 'mueller_v' and 'mueller_h' (each with length = 4).
+ Stokes vectors are xr.DataArray instances with dimensions that include 'stokes' (with length = 4).
+ The coordinate system is defined with its origin at the centre of the camera sensor, which lies in the xy plane.
+ S.I. units are used throughout, and all physical angles and phase delay angles are in radians.

Geometry definitions
--------------------

.. figure:: angles.pdf

Optical layout
--------------

List of modules
---------------

.. toctree::
   :maxdepth: 2
   :caption: Modules

   interferometer
   instrument
   camera
   coherence
   dispersion

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`









