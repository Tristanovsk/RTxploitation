"""
Inversion of in-situ Rrs spectra (Rogerio's dataset, M99 method) with the
bio-optical solver v2 (:mod:`case_study.mesodinium.solver_v2`).

Same as ``test_insitu.py`` but for the ``Rrs_stat.csv`` dataset; the median
Rrs (column ``'0.5'``) between 450 and 900 nm is inverted.

Input: ``/DATA/projet/rogerio/insitu/data/L2/all/Rrs_stat.csv``
Output: figures in ``figdir``
"""
import os
import numpy as np
import pandas as pd
import lmfit as lm

# ----------------------------
# set plotting styles
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

from RTxploitation import parameterization as RTp
import RTxploitation.auxdata as ad
import case_study.mesodinium as meso
from case_study.mesodinium.solver_v2 import Rrs_inversion

water = RTp.water()
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
# iop_file = 'gernez_IOPs_Mrubrum.txt'
iop_file = 'Ciotti_et_al_2002_aphy_star_Chl_18_34_41_Sf01.txt'
# load iop_star values
iops = pd.read_csv(opj(dataroot, iop_file), sep=' ', index_col=0)
if 'gernez' in iop_file:
    suffix = 'mesodinium'
else:
    suffix = 'Ciotti2002' + iop_file.split('_')[-1].replace('.txt', '')
    iops.columns = ['aphy_g', 'aphy_star']

# extend the specific IOPs up to 1100 nm with the last available values
additional_wl = np.arange(iops.index.values[-1] + 1, 1101, 1)
iops = pd.concat([iops, pd.DataFrame([iops.iloc[-1]] * len(additional_wl), index=additional_wl)])
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
xinit = [0.1, 2, 2, 0.015, 0.5]
os.makedirs(figdir, exist_ok=True)
for id, df_ in Rrss.groupby(level=[0,1]):
    print(id)
    df_ = df_[(df_.wl>450) & (df_.wl< 900)]
    Rrs = df_['0.5'].values
    wl = df_.wl


    solver = Rrs_inversion(a_star, bb_star, sza, wl_=wl)
    rrs = water.Rrs2rrs(Rrs)

    out = solver.call_solver(rrs, xinit=xinit)


    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    axes[0].plot(wl, Rrs, 'o-', label='data')
    axes[0].plot(wl, solver.forward_model(xinit, level='above'), 'k--', label='initial fit')
    axes[0].legend(loc='best')
    axes[1].plot(wl, Rrs, 'o-', label='data')
    axes[1].plot(wl, solver.forward_model(list(out.params.valuesdict().values()), level='above'), 'k--', label='retrieved')
    axes[1].legend(loc='best')
    axes[2].axis('off')
    axes[2].annotate(lm.fit_report(out),xy=(.0,.0),
        bbox=dict(boxstyle="round", fc='1'), fontsize=10)
    plt.suptitle(id[1] + ' / ' + str(id[0].date()))
    plt.savefig(opj(figdir, 'gordon_fit_' + id[1] + '_' + str(id[0].date()) + '.png'), dpi=200)
    plt.close()
