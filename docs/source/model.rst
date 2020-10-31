Model
=====

A framework for forward-modelling diagnostic images. Built using the `xarray library <http://xarray.pydata.org/en/stable/>`_ for labelled multi-dimensional arrays.

Conventions
-----------
- Mueller matrices are xr.DataArrays with dimensions that include 'mueller_v' and 'mueller_h' (each with length = 4).
- Stokes vectors are xr.DataArrays with dimensions that include 'stokes' (with length = 4).
- The coordinate system is defined with its origin at the centre of the camera sensor, which lies in the xy plane.
- S.I. units are used throughout, and all angles are in radians.

.. figure:: angles.pdf


.. toctree::
   :maxdepth: 2
   :caption: Modules

   interferometer
   instrument
   camera
   coherence
   dispersion

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`









