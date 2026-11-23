"""
Forward simulation and inversion test for *Mesodinium rubrum* blooms.

1. (optional, ``plot=True``) Forward modeling of Rrs spectra (Gordon et al.
   1988 + chlorophyll fluorescence) from the specific IOPs of
   *M. rubrum* for a range of chl-a concentrations, with and without
   a background component (CDOM + non-algal particles).
2. Closed-loop test of the inversion: Rrs simulated for several chl-a values
   are resampled at the Sentinel-2 MSI band centers and inverted with
   :class:`case_study.mesodinium.solver.Rrs_inversion`. The retrieved
   phytoplankton absorption and backscattering spectra are plotted.

Input: specific IOPs in ``data/`` (Gernez et al., M. rubrum)
Output: figures in ``figdir``
"""
import os
import numpy as np
import pandas as pd

# ----------------------------
# set plotting styles
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'font.size': 18})

from RTxploitation import parameterization as RTp
import RTxploitation.auxdata as ad
import case_study.mesodinium.plot_utils as uplot
from case_study.mesodinium.solver import Rrs_inversion

plot = False

opj = os.path.join
datadir = opj(os.path.dirname(os.path.abspath(__file__)), 'data')
figdir = '/DATA/projet/gernez/mesodinium/fig'

iops = pd.read_csv(opj(datadir, 'gernez_IOPs_Mrubrum.txt'), sep=' ', index_col=0)
# add coef for pure water
iops['aw'], iops['bbw'] = ad.iopw().get_iopw(iops.index)
iops_bg = pd.read_csv(opj(datadir, 'S2_hope_20170412_bgd_21-40.txt'), sep='\t', index_col=0)
iops_bg = iops_bg.to_xarray().interp(wl=iops.index).to_dataframe()

# spectrally flat backscattering for M. rubrum
iops['bbp_star'] = 4e-4

cmap = mpl.colormaps['Spectral'].reversed()
cmin, cmax = 0, 200
norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
wl = iops.index.values

water = RTp.water()

if plot:
    chls = np.logspace(np.log10(0.1), np.log10(200), 50)
    for coef in (0.5, 1, 2, 5):
        uplot.Rrs().plot_Rrs_a_bb(wl, iops.aphy_star, iops.bbp_star,
                                  iops.aw, iops.bbw, iops_bg.aphy + iops_bg.adg, iops_bg.bbp,
                                  param_star=chls, coef=coef, figdir=figdir)

# -----------------
# inversion algo
# -----------------
coef = 1
sza = 30
a_star, bb_star = iops.aphy_star.to_xarray(), iops.bbp_star.to_xarray()

# Sentinel-2 MSI band centers (nm), bands at 833 and 865 nm left out
wl_sat = np.array([443., 493, 560, 665, 704, 740, 783])
solver = Rrs_inversion(a_star, bb_star, sza, wl_=wl_sat)
# Ed at the fluorescence reference wavelength, same as used in the solver
Ed_ref = solver.Ed_ref

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
axs = axs.ravel()
chls = [1, 5, 10, 50, 75, 100, 200]
for chl in chls:
    bb = iops.bbw + coef * iops.bbp_star * chl
    a = iops.aw + iops.aphy_star * chl
    Rrs_g88 = water.gordon88(a, bb)
    Rrs_fluo = water.fluo_gower2004(chl, wl, Ed_ref=Ed_ref)
    Rrs_simu = Rrs_g88 + Rrs_fluo

    # resample to satellite bands and convert into below-water rrs
    Rrs_sat = Rrs_simu.to_xarray().interp(wl=wl_sat).values
    rrs_sat = water.Rrs2rrs(Rrs_sat)
    res = solver.call_solver(rrs_sat)
    chl_ = res.params['chl'].value
    print('chl simulated: {:.1f}, retrieved: {:.1f}'.format(chl, chl_))
    # '_' for retrieved parameters
    astar_ = chl_ * a_star
    bbstar_ = chl_ * bb_star

    axs[0].plot(wl, astar_, color=cmap(norm(chl_)), label='atot', lw=2.5, alpha=0.75)
    axs[1].plot(wl, bbstar_, color=cmap(norm(chl_)), label='bbtot', lw=2.5, alpha=0.75)

axs[0].set_ylabel(r'$a_{phy}\  (m^{-1})$')
axs[0].set_xlabel(r'Wavelength (nm)')
axs[1].set_ylabel(r'$b_{b_{phy}}\  (m^{-1})$')
axs[1].set_xlabel(r'Wavelength (nm)')
for i in [0, 1]:
    divider = make_axes_locatable(axs[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sm, cax=cax, format=mpl.ticker.ScalarFormatter(),
                 shrink=1.0, fraction=0.1, pad=0)

plt.suptitle('Retrieved phytoplankton IOPs (color: retrieved chl-a)')
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.86])
fig.subplots_adjust(left=0.1, right=0.9, hspace=.5, wspace=0.45)
fig.savefig(os.path.join(figdir, 'error_retrieval_a_bb_chl_compar.pdf'))

plt.show()
