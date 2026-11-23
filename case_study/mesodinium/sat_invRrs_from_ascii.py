"""
Inversion of satellite Rrs extracted at pixel level (SNAP PixEx ascii files)
with the ``invRrs`` package.

Same as ``sat_app_from_ascii.py`` but the inversion is done with the
:class:`invRrs.solver.two_modes` model (phytoplankton + background
absorption + two size modes of mineral particles with Mie-computed IOPs),
which retrieves: chl, a_bg_ref, S_bg, spm_norm, fine_prop, nr, ni_ref.

Input: PixEx ascii files (``satfiles``)
Output: ``<file>res_v2_<suffix>.csv`` in ``<dirdata>/satellite``
"""
import os
import glob
import numpy as np
import pandas as pd

import invRrs
import case_study.mesodinium as meso

opj = os.path.join
root = os.path.abspath(os.path.dirname(meso.__file__))
dataroot = opj(root, 'data')

project = 'theia'  # 'gernez' or 'theia'
if project == 'gernez':
    idir = '/DATA/projet/gernez/mesodinium'
    dirdata = opj(idir, 'data')
else:
    idir = '/DATA/projet/ces/theia'
    dirdata = opj(idir, 'fig')

sensor = 's2'
if sensor == 's2':
    if project == 'gernez':
        satfiles = glob.glob(opj(dirdata, 'satellite/raw', 'S2*.txt'))
    else:
        satfiles = glob.glob(opj(dirdata, 'pixEx*measurements.txt'))
    sat_props = ('s2a', list(range(1, 9)))
else:
    satfiles = glob.glob(opj(dirdata, 'satellite/raw', 'S3*.txt'))
    sat_props = ('s3a', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 20])

# ------------------------------
#   set iop parameters
# ------------------------------
# iop_file = 'gernez_IOPs_Mrubrum.txt'
iop_file = 'Ciotti_et_al_2002_aphy_star_Chl_18_34_41_Sf01.txt'
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

# viewing geometry used for the Rrs-IOP relationship coefficients
sza, vza, azi = 30, 5, 100

# ------------------------------
# unknowns and initial values
# ------------------------------
var_names = ['chl', 'a_bg_ref', 'S_bg', 'spm_norm', 'fine_prop', 'nr', 'ni_ref']
spm_norm = 10
nr, ni_ref = 1.17, -0.001
xinit = [10, 0.5, 0.017, spm_norm, 0.5, nr, ni_ref]

# ------------------------------
# load and loop on satellite
# pixel values
# ------------------------------
for satfile in satfiles:
    ofile = os.path.basename(satfile)
    satdata = pd.read_csv(satfile, skiprows=6, sep='\t')
    sza = satdata.SZA.mean()
    algo = invRrs.solver.two_modes(a_star, bb_star, sza, sat=sat_props[0], band_idx=sat_props[1])
    algo.set_coef(sza, vza, azi)

    # -----------------
    # initialization
    # -----------------
    satres = satdata.copy()
    for name in var_names:
        satres.loc[:, name] = np.nan

    if 'S3' in satfile:
        # OLCI reflectances are converted into Rrs
        Rrs = satdata.filter(regex='reflectance') / np.pi
    else:
        Rrs = satdata.filter(regex='Rrs_B[0-9]')
        Rrs = Rrs.iloc[:, sat_props[1]]

    # -----------------
    # perform fit
    # -----------------
    for idx, Rrs_ in Rrs.iterrows():
        out = algo.call_solver(Rrs_.values, xinit=xinit)
        if not out.success:
            continue
        satres.loc[idx, out.var_names] = out.x

    satres.to_csv(opj(dirdata, 'satellite', ofile.replace('.txt', 'res_v2_' + suffix + '.csv')), index=False)
