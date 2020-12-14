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
from study_cases.mesodinium.solver import Rrs_inversion

Rrs_inversion = meso.solver_v2.Rrs_inversion
water = RTp.water()
plot = False

opj = os.path.join
root = os.path.abspath(os.path.dirname(meso.__file__))
dataroot = opj(root, 'data')

idir = '/DATA/projet/gernez/mesodinium'
figdir = opj(idir, 'fig')
dirdata = opj(idir, 'data')
satfiles = glob.glob(opj(dirdata, 'satellite/S2*.txt'))


sensor='s3'

if sensor =='s2':
    sat_props = ('s2a', (range(9)))
else:
    sat_props = ('s3a', range(16))  # list([0,1,2,3,4,5,6,7,8,9,10,11,15]))#list([0,1,2,3,4,5,6,7,8,9,10,11,14]))

# ------------------------------
#   set iop parameters
# ------------------------------
iop_file = 'gernez_IOPs_Mrubrum.txt'  # 'Ciotti_et_al_2002_aphy_star_Chl_18_34_41_Sf01.txt'
# load iop_star values
if 'gernez' in iop_file:
    iops = pd.read_csv(opj(dataroot, iop_file), sep=' ', index_col=0)
    suffix = 'mesodinium'
else:
    iops = pd.read_csv(opj(dataroot, iop_file), sep=' ', index_col=0)
    suffix = 'Ciotti2002' + iop_file.split('_')[-1].replace('.txt', '')
    iops.columns = ['aphy_g', 'aphy_star']

# add wavelengths to cover full range up to 1000nm
additional_wl = np.arange(iops.index.values[-1] + 1, 1101, 1)
iops = iops.append(pd.DataFrame([iops.iloc[-1]] * len(additional_wl), index=additional_wl))
iops.index.name = 'wl'

# Strong assumption bbp spectrally fixed
iops['bbp_star'] = 4e-4

a_star, bb_star = iops.aphy_star.to_xarray(), iops.bbp_star.to_xarray()
sza = 30

nparams = 5  # number of parameters retrieved by solver
solver = Rrs_inversion(a_star, bb_star, sza, sat=sat_props[0], band_idx=sat_props[1])
var_names = ['chl', 'a_bg_ref', 'bb_bg_ref', 'S_bg', 'eta_bg']

# ------------------------------
# load satellite data
# ------------------------------
satfile = os.path.abspath('/home/harmel/satellite/s2/gernez/S3/' +
                          'subset_0_of_S3A_OL_2_WFR____20170413T105447_20170413T105647_20171108T040153_0119_016_265______MR1_R_NT_002.nc')
img = xr.open_dataset(satfile)
sza = img.SZA.data.mean()
keys = np.array(list(img.keys()))
cube = img.get(keys[sat_props[1]])
wls = []
for varname, da in cube.data_vars.items():
    wl = da.attrs['radiation_wavelength']
    print(wl)
    wls.append(wl)
cube = cube.to_array(dim='wavelength', name='Rrs')
cube.wavelength.data = wls
w_, x_, y_ = cube.shape
arr = np.array(cube.data) / np.pi
res = np.zeros((nparams, x_, y_))

# ------------------------------
# load and loop on satellite
# pixel values
# ------------------------------
wl_min = 0
wl_max = 800

for ix in range(x_):
    for iy in range(y_):
        Rrs = arr[:, ix, iy]
        if (Rrs >= 0).all():
            print(Rrs)
            rrs = water.Rrs2rrs(Rrs)
            res[:, ix, iy] = solver.call_solver(rrs).x

cube.data[:5, ...] = res
cube.to_netcdf(os.path.basename(satfile).replace('.nc', '') + suffix + '.nc')

#
#
# satdata = pd.read_csv(satfile, skiprows=6, sep='\t')
# wl = pd.read_csv(satfile, skiprows=5, nrows=1, index_col=0, header=None, sep='\t')
# wl = wl[(wl > 0) & (wl < 1100)]
# wl = wl.dropna(axis=1)
#
# wl_sat = np.unique(wl)
# idx_wl = (wl_sat > wl_min) & (wl_sat < wl_max)
# sza = satdata.SZA.mean()
#
# # -----------------
# # initialization
# # -----------------
#
# satres = satdata.copy()
# for name in var_names:
#     satres.loc[:, name] = 0
#
# if 'S3' in satfile:
#     Rrs = satdata.filter(regex='reflectance')
#     wl_ = wl_sat[idx_wl]
#     Rrs = Rrs.iloc[:, idx_wl]
#
# else:
#     Rrs = satdata.filter(regex='Rrs_B[0-9]')
#     # remove SWIR bands
#     Rrs = Rrs.drop(['Rrs_B11', 'Rrs_B12'], axis=1)
#     # for tests remove other bands
#     Rrs = Rrs.drop(['Rrs_B1', 'Rrs_B8A'], axis=1)#'Rrs_B8',
#     wl_ = wl_sat[[range(1, 7)]]
#
# # a_star_sat = set_wl(a_star, wl_)
# # bb_star_sat = set_wl(bb_star, wl_)
# # solver = Rrs_inversion(wl_, a_star_sat, bb_star_sat, sza)
#
# # -----------------
# # inversion algo
# # -----------------
#
# # transform to 0- (subsurface) rrs
# rrs = water.Rrs2rrs(Rrs)
# res = rrs.apply(solver.call_solver, axis=1)
# # res = rrs.parallel_apply(solver.call_solver, axis=1)
# # res = solver.multiprocess(rrs)
#
# # -----------------
# # # save results
# # -----------------
# for idx, x in enumerate(res):
#     satres.loc[idx, x.var_names] = x.x
#
# satres.to_csv(satfile.replace('.txt', 'res_v2_CiottiSf01.csv'), index_label=False)
