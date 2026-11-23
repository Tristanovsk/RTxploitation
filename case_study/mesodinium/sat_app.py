"""
Pixel-by-pixel inversion of a satellite Rrs image (netCDF subset).

Each pixel of the image is inverted with the bio-optical solver v2
(:mod:`case_study.mesodinium.solver_v2`) using the band-averaged specific
IOPs of *M. rubrum* (or Ciotti et al. 2002 phytoplankton, see ``iop_file``).
The retrieved parameters (chl, a_bg_ref, bb_bg_ref, S_bg, eta_bg) are stored
in the first five layers of the output cube.

Input: Sentinel-3 OLCI (or Sentinel-2 MSI) subset in netCDF (``satfile``)
Output: netCDF file ``<satfile basename><suffix>.nc`` in ``odir``
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

from RTxploitation import parameterization as RTp
import case_study.mesodinium as meso
from case_study.mesodinium.solver_v2 import Rrs_inversion

water = RTp.water()

opj = os.path.join
root = os.path.abspath(os.path.dirname(meso.__file__))
dataroot = opj(root, 'data')

idir = '/DATA/projet/gernez/mesodinium'
odir = opj(idir, 'data', 'satellite')
satfile = opj('/home/harmel/satellite/s2/gernez/S3/',
              'subset_0_of_S3A_OL_2_WFR____20170413T105447_20170413T105647_20171108T040153_0119_016_265______MR1_R_NT_002.nc')

sensor = 's3'
if sensor == 's2':
    sat_props = ('s2a', list(range(9)))
else:
    sat_props = ('s3a', list(range(16)))

# ------------------------------
#   set iop parameters
# ------------------------------
iop_file = 'gernez_IOPs_Mrubrum.txt'  # 'Ciotti_et_al_2002_aphy_star_Chl_18_34_41_Sf01.txt'
# load iop_star values
iops = pd.read_csv(opj(dataroot, iop_file), sep=' ', index_col=0)
if 'gernez' in iop_file:
    suffix = 'mesodinium'
else:
    suffix = 'Ciotti2002' + iop_file.split('_')[-1].replace('.txt', '')
    iops.columns = ['aphy_g', 'aphy_star']

# add wavelengths to cover full range up to 1100 nm
additional_wl = np.arange(iops.index.values[-1] + 1, 1101, 1)
iops = pd.concat([iops, pd.DataFrame([iops.iloc[-1]] * len(additional_wl), index=additional_wl)])
iops.index.name = 'wl'

# Strong assumption bbp spectrally fixed
iops['bbp_star'] = 4e-4

a_star, bb_star = iops.aphy_star.to_xarray(), iops.bbp_star.to_xarray()

# ------------------------------
# load satellite data
# ------------------------------
img = xr.open_dataset(satfile)
sza = float(img.SZA.data.mean())
nparams = 5  # number of parameters retrieved by solver
solver = Rrs_inversion(a_star, bb_star, sza, sat=sat_props[0], band_idx=sat_props[1])
var_names = ['chl', 'a_bg_ref', 'bb_bg_ref', 'S_bg', 'eta_bg']

keys = np.array(list(img.keys()))
cube = img.get(keys[sat_props[1]])
wls = []
for varname, da in cube.data_vars.items():
    wls.append(da.attrs['radiation_wavelength'])
cube = cube.to_array(dim='wavelength', name='Rrs').assign_coords(wavelength=wls)
w_, x_, y_ = cube.shape
# OLCI reflectances are converted into Rrs
arr = np.array(cube.data) / np.pi
res = np.zeros((nparams, x_, y_))

# ------------------------------
# loop on satellite pixel values
# ------------------------------
for ix in range(x_):
    for iy in range(y_):
        Rrs = arr[:, ix, iy]
        if (Rrs >= 0).all():
            rrs = water.Rrs2rrs(Rrs)
            res[:, ix, iy] = solver.call_solver(rrs).x

cube.data[:nparams, ...] = res
cube.to_netcdf(opj(odir, os.path.basename(satfile).replace('.nc', '') + suffix + '.nc'))
