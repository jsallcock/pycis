import setuptools

setuptools.setup(
    name='pycis',
    version='0.1',
    author='Joseph Allcock',
    description='Analysis and modelling for the Coherence Imaging Spectroscopy (CIS) plasma diagnostic',
    url='https://github.com/jsallcock/pycis',
    install_requires=['numpy', 'scipy', 'matplotlib', 'xarray', 'numba'],
    packages=setuptools.find_packages(),
)
