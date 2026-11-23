"""
RTxploitation
=============

Tools to simulate, store, exploit and visualize radiative transfer (RT)
related parameters, mainly from OSOAA (Ocean Successive Orders with
Atmosphere Advanced) vector radiative transfer simulations.

Main modules
------------

- :mod:`RTxploitation.auxdata`: ancillary data (pure water IOPs, solar
  irradiance, CDOM absorption).
- :mod:`RTxploitation.parameterization`: simple atmospheric and bio-optical
  parameterizations (downwelling irradiance, Gordon et al. 1988 model,
  chlorophyll fluorescence).
- :mod:`RTxploitation.utils`: utilities to build netCDF look-up tables (LUT)
  from OSOAA outputs.
- :mod:`RTxploitation.spectral_integration`: satellite relative spectral
  responses and band convolution.
- :mod:`RTxploitation.psd`: particle size distributions and their
  geometrical parameters.
- :mod:`RTxploitation.lutplot`: polar diagram plotting helpers.
- :mod:`RTxploitation.load_lut_nc`: loader of OSOAA netCDF LUT files into
  :class:`xarray.Dataset`.
- :mod:`RTxploitation.list_parameter_lut`: containers for LUT metadata.
- :mod:`RTxploitation.sunglint.cox_munk`: Cox-Munk sunglint model.
"""

__all__ = ['utils','lutplot',]

from .version import __version__
from . import auxdata, parameterization, utils, spectral_integration, version
