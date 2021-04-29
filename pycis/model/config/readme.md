pycis/model/config/
===================

.yaml config files can be used to quickly instantiate pycis.Instrument.

- This directory is searched for a matching .yaml config files when pycis.Instrument is instantiated with the kwarg 
  'config'.

- Example config files are provided.

- You don't have to use this directory at all, since config files can be linked using an absolute path or 
  the pycis.Instrument config can be specified using pure Python. See docs / docstring.

