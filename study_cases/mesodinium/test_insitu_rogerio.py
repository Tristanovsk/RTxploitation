import os, sys
import glob
import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
from pandarallel import pandarallel
import lmfit as lm
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


water = RTp.water()
plot = False

opj = os.path.join
root = os.path.abspath(os.path.dirname(meso.__file__))
dataroot = opj(root, 'data')

idir = '/DATA/projet/rogerio'
figdir = opj(idir, 'fig',)
dirdata = opj(idir,'insitu/data/L2/all')

# ---------------------------
# load Rrs data
# ---------------------------

df = pd.read_csv(opj(dirdata, 'Rrs_stat.csv') , index_col=[0, 1], parse_dates=True)
Rrss = df[df.method=='M99']

# ------------------------------
#   set iop parameters
# ------------------------------
iop_file = 'gernez_IOPs_Mrubrum.txt' #
iop_file =  'Ciotti_et_al_2002_aphy_star_Chl_18_34_41_Sf01.txt'
# load iop_star values
if 'gernez' in iop_file:
    iops = pd.read_csv(opj(dataroot, iop_file), sep=' ', index_col=0)
    suffix='mesodinium'
else:
    iops = pd.read_csv(opj(dataroot,iop_file ), sep=' ', index_col=0)
    suffix='Ciotti2002'+iop_file.split('_')[-1].replace('.txt','')
    iops.columns = ['aphy_g','aphy_star']

additional_wl = np.arange(iops.index.values[-1] + 1, 1101, 1)
iops = iops.append(pd.DataFrame([iops.iloc[-1]] * len(additional_wl), index=additional_wl))
iops.index.name = 'wl'

# Strong assumption bbp spectrally fixed
iops['bbp_star'] = 4e-4

a_star, bb_star = iops.aphy_star.to_xarray(), iops.bbp_star.to_xarray()
# add coef for pure water
iops['aw'], iops['bbw'] = ad.iopw().get_iopw(iops.index)
wl = iops.index.values


# ------------------------------
# Process spectra
# ------------------------------

sza = 30
var_names = ['chl', 'a_bg_ref', 'bb_bg_ref', 'S_bg', 'eta_bg']
xinit=[0.1,2,2,0.015,0.5]
for id, df_ in Rrss.groupby(level=[0,1]):
    print(id)
    df_ = df_[(df_.wl>400) & (df_.wl< 900)]
    Rrs = df_['0.5'].values
    wl = df_.wl


    solver = Rrs_inversion(a_star, bb_star, sza, wl_=wl)
    rrs = water.Rrs2rrs(Rrs)

    out = solver.call_solver(rrs,xinit=xinit)


    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    axes[0].plot(wl, Rrs, 'o-', label='data')
    axes[0].plot(wl, solver.forward_model(xinit,level='above'), 'k--', label='initial fit')
    axes[0].legend(loc='best')
    axes[1].plot(wl, Rrs, 'o-', label='data')
    axes[1].plot(wl, solver.forward_model(out.x,level='above'), 'k--', label='retrieved')
    axes[1].legend(loc='best')
    axes[2].axis('off')
    axes[2].annotate(lm.fit_report(out),xy=(.0,.0),
        bbox=dict(boxstyle="round", fc='1'), fontsize=10)
    plt.suptitle(id[1]+' / '+id[0].date().__str__())
    plt.savefig(opj(figdir,'gordon_fit_'+id[1]+'_'+id[0].date().__str__()),dpi=200)
    plt.close()