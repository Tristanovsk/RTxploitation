import os, sys
import glob
import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=10)

# ----------------------------
# set plotting styles
import cmocean as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# plt.ioff()
plt.rcParams.update({'font.size': 18})

from RTxploitation import lutplot
from RTxploitation import utils as u
from RTxploitation import parameterization as RTp
import RTxploitation.auxdata as ad
import study_cases.mesodinium.plot_utils as uplot
import study_cases.mesodinium as meso
from study_cases.mesodinium.solver_v2 import Rrs_inversion

import invRrs

water = RTp.water()
plot = False

opj = os.path.join
root = os.path.abspath(os.path.dirname(meso.__file__))
dataroot = opj(root, 'data')

idir = '/DATA/projet/gernez/mesodinium'
figdir = opj(idir, 'fig')
dirdata = opj(idir, 'data')
idir = '/DATA/projet/ces/theia'
figdir = opj(idir, 'fig')
dirdata = opj(idir, 'fig')
sensor = 's2'

if sensor == 's2':
    satfiles = glob.glob(opj(dirdata, 'satellite/raw', 'S2*.txt'))
    satfiles = glob.glob(opj(dirdata, 'pixEx*measurements.txt'))
    sat_props = ('s2a', (range(1, 9)))
else:
    satfiles = glob.glob(opj(dirdata, 'satellite/raw', 'S3*.txt'))
    sat_props = (
    's3a', list([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 20]))  # list([0,1,2,3,4,5,6,7,8,9,10,11,14]))

# ------------------------------
#   set iop parameters
# ------------------------------
iop_file = 'gernez_IOPs_Mrubrum.txt'  #
iop_file = 'Ciotti_et_al_2002_aphy_star_Chl_18_34_41_Sf01.txt'
# load iop_star values
if 'gernez' in iop_file:
    iops = pd.read_csv(opj(dataroot, iop_file), sep=' ', index_col=0)
    suffix = 'mesodinium'
else:
    iops = pd.read_csv(opj(dataroot, iop_file), sep=' ', index_col=0)
    suffix = 'Ciotti2002' + iop_file.split('_')[-1].replace('.txt', '')
    iops.columns = ['aphy_g', 'aphy_star']

additional_wl = np.arange(iops.index.values[-1] + 1, 1101, 1)
iops = iops.append(pd.DataFrame([iops.iloc[-1]] * len(additional_wl), index=additional_wl))
iops.index.name = 'wl'

# Strong assumption bbp spectrally fixed
iops['bbp_star'] = 4e-4

a_star, bb_star = iops.aphy_star.to_xarray(), iops.bbp_star.to_xarray()
sza = 30
vza = 5
azi = 100

algo = invRrs.solver.two_modes(a_star, bb_star, sza, sat=sat_props[0], band_idx=sat_props[1])
algo.set_g1g2(sza, vza, azi)

# add coef for pure water
iops['aw'], iops['bbw'] = ad.iopw().get_iopw(iops.index)
wl = iops.index.values

var_names = ['chl', 'a_bg_ref', 'bb_bg_ref', 'S_bg', 'eta_bg']


# convert into satellite bands
def set_wl(df, wl):
    return df.to_xarray().interp(wl=wl).to_pandas()


cmap = plt.cm.get_cmap("Spectral").reversed()
decimal = 3
cmin, cmax = 0, 200  # (dff[param].min() * 0.8).round(decimal), (dff[param].max() * 1.2).round(decimal)

norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# ------------------------------
# load and loop on satellite
# pixel values
# ------------------------------
var_names = ['chl', 'a_bg_ref', 'S_bg', 'spm_fine', 'spm_coarse', 'nr_ref', 'ni_ref', 'S_ni']
spm_norm = 10
nr, ni_ref, S_ni = 1.17, -0.001, 0.005
xinit = [10, 0.5, 0.017, 0.5 * spm_norm, 0.5 * spm_norm, nr, ni_ref, 0.005]  # [0.1,2,2,0.015,0.5]
res = []
wl_min, wl_max = 400, 900

for satfile in satfiles:
    ofile = os.path.basename(satfile)
    satdata = pd.read_csv(satfile, skiprows=6, sep='\t')
    wl = pd.read_csv(satfile, skiprows=5, nrows=1, index_col=0, header=None, sep='\t', na_values=' ')

    wl = wl[wl > 0].dropna(axis=1)
    wl_sat = np.unique(wl)
    sza = satdata.SZA.mean()

    # -----------------
    # initialization
    # -----------------

    satres = satdata.copy()
    for name in var_names:
        satres.loc[:, name] = 0

    if 'S3' in satfile:
        Rrs = satdata.filter(regex='reflectance') / np.pi
        wl_ = wl_sat
    else:
        Rrs = satdata.filter(regex='Rrs_B[0-9]')
        wl_ = wl_sat[sat_props[1]]
        Rrs = Rrs.iloc[:, sat_props[1]]

    # -----------------
    # inversion algo
    # -----------------
    # perform fit
    for idx, Rrs_ in Rrs.iterrows():
        print(idx)
        out = algo.call_solver(Rrs_.values, xinit=xinit)
        if not out.success:
            continue
        x = out.x
        satres.loc[idx, var_names] = x

    satres.to_csv(opj(dirdata, 'satellite', ofile.replace('.txt', 'res_v2_' + suffix + '.csv')), index=False)
    # satres.to_csv(opj(dirdata,ofile.replace('.txt', 'res_v2_'+suffix+'.csv')), index_label=False)
