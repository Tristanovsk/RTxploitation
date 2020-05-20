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
from study_cases.mesodinium.solver import Rrs_inversion

plot = False

opj = os.path.join
idir = '/DATA/projet/gernez/mesodinium'
datadir = opj(idir, 'data')
figdir = opj(idir, 'fig')
satfiles = glob.glob(opj(datadir, 'satellite/*GRS.txt'))

# ------------------------------
#   set iop parameters
# ------------------------------

# load iop_star values
iops = pd.read_csv(opj(datadir, 'Art1_Fig4_IOPs_Mrubrum.txt'), sep=' ', index_col=0)
# add wavelengths to cover full range up to 1000nm
additional_wl = np.arange(iops.index.values[-1] + 1, 1001, 1)
iops = iops.append(pd.DataFrame([iops.iloc[-1]] * len(additional_wl), index=additional_wl))
iops.index.name = 'wl'
water = RTp.water()
iops.bbp_star = 4e-4
a_star, bb_star = iops.aphy_star, iops.bbp_star

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

for satfile in satfiles:

    satdata = pd.read_csv(satfile, skiprows=6, sep='\t')
    wl = pd.read_csv(satfile, skiprows=5, nrows=1, index_col=0, header=None, sep='\t')
    wl = wl[(wl > 0) & (wl < 1000)]
    wl = wl.dropna(axis=1)

    wl_sat = np.unique(wl)
    sza = satdata.SZA.mean()



    # -----------------
    # initialization
    # -----------------

    satres = satdata.copy()
    for name in var_names:
        satres.loc[:, name] = 0

    Rrs = satdata.filter(regex='Rrs_B[0-9]')
    # remove SWIR bands
    Rrs = Rrs.drop(['Rrs_B11', 'Rrs_B12'], axis=1)
    # for tests remove other bands
    Rrs = Rrs.drop(['Rrs_B1','Rrs_B8', 'Rrs_B8A'], axis=1)
    wl_ = wl_sat[[range(1,7)]]
    a_star_sat = set_wl(a_star, wl_)
    bb_star_sat = set_wl(bb_star, wl_)
    solver = Rrs_inversion(wl_, a_star_sat, bb_star_sat, sza)

    # -----------------
    # inversion algo
    # -----------------

    # transform to 0- (subsurface) rrs
    rrs = water.Rrs2rrs(Rrs)
    #res = rrs.apply(solver.call_solver, axis=1)
    res = rrs.parallel_apply(solver.call_solver, axis=1)
    # res = solver.multiprocess(rrs)

    # -----------------
    # # save results
    # -----------------
    for idx, x in enumerate(res):
        satres.loc[idx, x.var_names] = x.x

    satres.to_csv(satfile.replace('.txt', 'res.csv'), index_label=False)
