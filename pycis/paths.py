import os, inspect

"""
The saved filter bandpass ci_data_mast will probably be moved elsewhere, but for now, the saved filter bandpass ci_data_mast can be 
accessed using pycis.model.FilterFromName or else directly using pycis.paths.filters_path

"""

root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))