"""
Inversion of in-situ Rrs spectra (TriOS, Bagre 2015 turbid waters) with the
bio-optical solver v2 (:mod:`case_study.mesodinium.solver_v2`).

For each spectrum, Rrs is inverted into chl-a and background IOPs using the
specific absorption of the selected algae (``iop_file``) and a spectrally
flat bbp*. One figure per spectrum compares data, initial guess and fit, with
the lmfit report.

Input: ``data/QC_trios_Rrs_turbid_bagre2015.csv``
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

idir = '/DATA/projet/gernez/mesodinium'
figdir = opj(idir, 'fig','bagre')
dirdata = opj(idir,'data')

# ---------------------------
# load Rrs data
# ---------------------------

def reader(file, header=[0], index_col=[0, 1]):
    '''
    Read into multiindex pandas format
    :param file:
    :param header: array of the row numbers to be used for multiindex columns
    :return:
    '''

    df = pd.read_csv(file, index_col=index_col, header=header, parse_dates=True)
    if len(header) > 1:
        for i, columns_old in enumerate(df.columns.levels):
            columns_new = np.where(columns_old.str.contains('Unnamed'), '', columns_old)
            df = df.rename(columns=dict(zip(columns_old, columns_new)), level=i)
    return df

df = reader(opj(dataroot, 'QC_trios_Rrs_turbid_bagre2015.csv'), index_col=[0, 1, 2, 3], header=[0, 1])
df['IDnum'] = df.index.get_level_values(0).str.extract(r'(\d+)').values.astype('int')
df = df.reset_index()
new_dates, new_times = zip(*[(d.date(), d.time()) for d in df['date']])
df = df.assign(date=new_dates, time=new_times)
df = df.set_index(['ID', 'date'])


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
iops['bbp_star'] = 4e-3

a_star, bb_star = iops.aphy_star.to_xarray(), iops.bbp_star.to_xarray()
# add coef for pure water
iops['aw'], iops['bbw'] = ad.iopw().get_iopw(iops.index)
wl = iops.index.values


# ------------------------------
# Process spectra
# ------------------------------


var_names = ['chl', 'a_bg_ref', 'bb_bg_ref', 'S_bg', 'eta_bg']
xinit = [0.1, 2, 2, 0.015, 0.5]
os.makedirs(figdir, exist_ok=True)
for id, df_ in df.iterrows():
    print(id)
    Rrs = df_.Rrs_osoaa_coarse.astype('float32')
    wl = Rrs.index.astype('float32')
    sza = float(df_.sza.iloc[0])
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
    plt.suptitle(id[0] + ' / ' + str(id[1]))
    plt.savefig(opj(figdir, 'gordon_fit_bagre_' + id[0] + '_' + str(id[1]) + '.png'), dpi=200)
    plt.close()
