import os, sys
import glob
import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
from scipy import odr as odr

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

iops = pd.read_csv(opj(datadir, 'Art1_Fig4_IOPs_Mrubrum.txt'), sep=' ', index_col=0)
# add coef for pure water
iops['aw'], iops['bbw'] = ad.iopw().get_iopw(iops.index)
iops_bg = pd.read_csv(opj(datadir, 'S2_hope_20170412_bgd_21-40.txt'), sep='\t', index_col=0)
iops_bg = iops_bg.to_xarray().interp(wl=iops.index).to_dataframe()

chls = np.logspace(np.log10(0.1), np.log10(200), 50)

title = 'Chl --> Rrs (model direct)'
cmap = plt.cm.get_cmap("Spectral").reversed()

decimal = 3
cmin, cmax = 0, 200  # (dff[param].min() * 0.8).round(decimal), (dff[param].max() * 1.2).round(decimal)

norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
wl = iops.index.values

water = RTp.water()
iops.bbp_star = 4e-4

if plot:
    for coef in (0.5, 1, 2, 5):
        uplot.Rrs().plot_Rrs_a_bb(wl, iops.aphy_star, iops.bbp_star,
                                  iops.aw, iops.bbw, iops_bg.aphy + iops_bg.adg, iops_bg.bbp,
                                  param_star=chls, coef=coef, figdir=figdir)

# -----------------
# inversion algo
# -----------------
coef = 1
chl = 50
sza = 30
a_star, bb_star = iops.aphy_star, iops.bbp_star

# convert into satellite bands
def set_wl(df,wl):
    return df.to_xarray().interp(wl=wl).to_pandas()

wl_sat = np.array([443., 493, 560, 665, 704, 740, 783]) #, 833, 865])
a_star_sat = set_wl(a_star,wl_sat)
bb_star_sat = set_wl(bb_star,wl_sat)

solver = Rrs_inversion(wl_sat)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))  # ncols=3, figsize=(30, 12))
axs = axs.ravel()
chls=[1,5,10,50,75,100,200]
for chl in chls:
    bb = iops.bbw + coef * iops.bbp_star * chl
    a = iops.aw + iops.aphy_star * chl
    Rrs_g88 = water.gordon88(a, bb)
    Rrs_fluo = water.fluo_gower2004(chl, wl, sza)
    Rrs_simu = Rrs_g88 + Rrs_fluo
    # '_' for retrieved parameters


    Rrs_sat = Rrs_simu.to_xarray().interp(wl=wl_sat).to_pandas()
    a_, bb_ = solver.Rrs_to_iops(Rrs_sat)
    res=solver.call_solver(Rrs_sat, a_star_sat, bb_star_sat,sza)
    x=res.x
    astar_ = x[0]* a_star
    bbstar_ = x[0]* bb_star

    axs[0].plot(wl, astar_, color=cmap(norm(x[0])), label='atot', lw=2.5, alpha=0.75)
    axs[1].plot(wl, bbstar_, color=cmap(norm(x[0])), label='bbtot', lw=2.5, alpha=0.75)

axs[0].set_ylabel(r'$\Delta a_{tot}\  (m^{-1})$')
axs[0].set_xlabel(r'Wavelength (nm)')
axs[1].set_ylabel(r'$\Delta b_{b_{tot}}\  (m^{-1})$')
axs[1].set_xlabel(r'Wavelength (nm)')
for i in [0, 1]:
    divider = make_axes_locatable(axs[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sm, cax=cax, format=mpl.ticker.ScalarFormatter(),
                        shrink=1.0, fraction=0.1, pad=0)

plt.suptitle('Error in inversion')
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.86])
fig.subplots_adjust(left=0.1, right=0.9, hspace=.5, wspace=0.45)
fig.savefig(os.path.join(figdir, 'error_retrieval_a_bb_chl_compar.pdf'))

plt.show()
