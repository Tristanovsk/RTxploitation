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
from pandarallel import pandarallel

plt.ioff()
plt.rcParams.update({'font.size': 18})

from RTxploitation import lutplot
from RTxploitation import utils as u
from RTxploitation import parameterization as RTp
import RTxploitation.auxdata as ad
import study_cases.mesodinium.plot_utils as uplot
from study_cases.mesodinium.solver_v2 import Rrs_inversion
import study_cases.mesodinium as meso

plot = False

opj = os.path.join
root = os.path.abspath(os.path.dirname(meso.__file__))
dataroot = opj(root, 'data')


idir = '/DATA/projet/gernez/mesodinium'
figdir = opj(idir, 'fig')
dirdata = opj(idir,'data')

sensor='s2'

if sensor =='s2':
    satfiles = glob.glob(opj(dirdata, 'satellite/S2*res_v2*.csv'))
    sat_props=('s2a',list(range(9)))
    wl_max=900
else:
    satfiles = glob.glob(opj(dirdata, 'satellite/S3*res_v2*.csv'))
    sat_props=('s3a',list([0,1,2,3,4,5,6,7,8,9,10,11,15,16,17,20])) #list([0,1,2,3,4,5,6,7,8,9,10,11,14]))
    wl_max=1020

# load iop_star values
iops = pd.read_csv(opj(dataroot, 'gernez_IOPs_Mrubrum.txt'), sep=' ', index_col=0)
# add wavelengths to cover full range up to 1000nm
additional_wl = np.arange(iops.index.values[-1] + 1, 1101, 1)
iops = iops.append(pd.DataFrame([iops.iloc[-1]] * len(additional_wl), index=additional_wl))
iops.index.name = 'wl'

iops.bbp_star = 4e-4

a_star, bb_star = iops.aphy_star.to_xarray(), iops.bbp_star.to_xarray()
sza=30

solver = Rrs_inversion(a_star, bb_star, sza, sat=sat_props[0], band_idx=sat_props[1])

# load satellite pixel values

cmap = plt.cm.get_cmap("Spectral").reversed()

decimal = 3
cmin, cmax = 0, 200  # (dff[param].min() * 0.8).round(decimal), (dff[param].max() * 1.2).round(decimal)

norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

wl_width = []
wl_sat = solver.wl
wl = iops.index.values

water = RTp.water()
#iops.bbp_star = 8e-4

# -----------------
# plotting results
# -----------------

for satfile in satfiles:
    title = '_'.join(satfile.split(r'_')[-6:]).replace('.csv', '')
    title = os.path.basename(satfile).replace('.csv', '')

    print(title)
    df = pd.read_csv(satfile)
    sza = df.SZA.mean()

    solver = Rrs_inversion(a_star, bb_star,sza,sat=sat_props[0], band_idx=sat_props[1] )
    solver_hyp = Rrs_inversion( a_star, bb_star,sza,wl_=wl)

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(21, 21))  # ncols=3, figsize=(30, 12))
    for irow, chl in  enumerate(([0,20],[20,40],[40,1000])):
        print(irow)
        norm = mpl.colors.Normalize(vmin=chl[0], vmax=min(100,chl[1]))
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        df_ = df[(df.chl>=chl[0])&(df.chl<chl[1])]

        if sensor == 's2':
            Rrs_sat = df_.filter(regex='Rrs_B[0-9]').astype('float')
        else:
            Rrs_sat = df_.filter(regex='reflectance').astype('float')/np.pi

        # remove SWIR bands:
        Rrs_ = Rrs_sat.iloc[:,:-2].values
        wl_ = [wl_sat] * Rrs_.shape[0]
        xy_max = Rrs_.max() * 1.2 #(max(max(Rrs_sat), max(Rrs_est)) * 1.2)

        for idx, sat in df_.iterrows():
            #Rrs_sat = df_.filter(regex='reflectance').astype('float')
            if sensor == 's2':
                Rrs_sat = sat.filter(regex='Rrs_B[0-9]').astype('float')
                # get appropriate bands:
                Rrs_sat = Rrs_sat[sat_props[1]]
            else:
                Rrs_sat = sat.filter(regex='reflectance').astype('float')/np.pi


            x = sat.iloc[-5:]

            # satellite data
            axs[irow,0].plot(wl_sat, Rrs_sat, color=cmap(norm(x[0])), label='Measured', lw=2.5, alpha=0.75)

            # hyperspectral simulations
            Rrs_est = solver_hyp.forward_model(x,level='above')
            axs[irow,1].plot(wl, Rrs_est, '--', color=cmap(norm(x[0])), label='Estimation', lw=2.5, alpha=0.75)

            # comparison simu/data
            Rrs_est = solver.forward_model(x,level='above')
            compar =axs[irow,2].scatter(Rrs_sat, Rrs_est, c=wl_sat,cmap='Spectral_r', alpha=0.6)

        axs[irow,0].set_ylabel(r'$Rrs\ (sr^{-1})$')
        axs[irow,0].set_xlabel(r'Wavelength (nm)')
        axs[irow,0].set_title(str(chl[0])+r'$ \leq chl <$'+str(chl[1])+', Satellite')
        axs[irow,0].set_xlim(400, wl_max)
        axs[irow,0].set_ylim(-2e-3,xy_max)
        axs[irow,1].set_ylabel(r'$Rrs\ (sr^{-1})$')
        axs[irow,1].set_xlabel(r'Wavelength (nm)')
        axs[irow,1].set_title(r'Reconstructed')
        axs[irow,1].set_xlim(400, wl_max)
        axs[irow,1].set_ylim(-2e-3,xy_max)
        axs[irow,2].set_ylabel(r'$Rrs_{modeled}$')
        axs[irow,2].set_xlabel(r'$Rrs_{satellite}$')
        axs[irow,2].set_title(r'Comparison')
        axs[irow,2].set_xlim(-2e-3,xy_max)
        axs[irow,2].set_ylim(-2e-3,xy_max)
        axs[irow,2].plot([-100, 100], [-100, 100], 'k--', lw=2)
        divider = make_axes_locatable(axs[irow,2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(compar,cax=cax)

        for i in [0, 1]:
            divider = make_axes_locatable(axs[irow,i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(sm, cax=cax, format=mpl.ticker.ScalarFormatter(),
                                shrink=1.0, fraction=0.1, pad=0)

    plt.suptitle(title)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.86])
    fig.subplots_adjust(left=0.1, right=0.9, hspace=.5, wspace=0.45)
    fig.savefig(os.path.join(figdir, 'Rrs_retieval_compar_' + title + '_v2.png'),dpi=200)
    plt.close()
    data = df.iloc[:, -5:]
    data.hist()
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.86])
    #plt.show()


