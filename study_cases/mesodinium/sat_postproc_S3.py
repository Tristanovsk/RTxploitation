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
from study_cases.mesodinium.solver import Rrs_inversion

plot = False

opj = os.path.join
idir = '/DATA/projet/gernez/mesodinium'
datadir = opj(idir, 'data')
figdir = opj(idir, 'fig')

# load iop_star values
iops = pd.read_csv(opj(datadir, 'Art1_Fig4_IOPs_Mrubrum.txt'), sep=' ', index_col=0)
# add wavelengths to cover full range up to 1000nm
additional_wl = np.arange(iops.index.values[-1] + 1, 1001, 1)
iops = iops.append(pd.DataFrame([iops.iloc[-1]] * len(additional_wl), index=additional_wl))
iops.index.name = 'wl'

# add coef for pure water
iops['aw'], iops['bbw'] = ad.iopw().get_iopw(iops.index)

# load satellite pixel values
satfiles = glob.glob(opj(datadir, 'satellite/*GRS.txt'))
cmap = plt.cm.get_cmap("Spectral").reversed()

decimal = 3
cmin, cmax = 0, 200  # (dff[param].min() * 0.8).round(decimal), (dff[param].max() * 1.2).round(decimal)

norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

wl_S2 = np.array([442.7316, 492.441, 559.8538, 664.6208, 704.1223, 740.4838,
                   782.751, 832.77, 864.7027])
wl_S3= np.array([ 400.  ,  412.5 ,  442.5 ,  490.  ,  510.  ,  560.  ,  620.  ,
        665.  ,  673.75,  681.25,  708.75,  753.75,  778.75,  865.  ,
        885.  , 1020.  ])
wl_width = []
wl_sat =wl_S3
wl = iops.index.values

water = RTp.water()
iops.bbp_star = 8e-4

# -----------------
# inversion algo
# -----------------
coef = 1
chl = 50

a_star, bb_star = iops.aphy_star, iops.bbp_star


# convert into satellite bands
def set_wl(df, wl):
    return df.to_xarray().interp(wl=wl).to_pandas()


var_names = ['chl', 'a_bg_ref', 'bb_bg_ref', 'S_bg', 'eta_bg']
a_star_sat = set_wl(a_star, wl_sat)
bb_star_sat = set_wl(bb_star, wl_sat)

# load satellite pixel results
satfiles = glob.glob(opj(datadir, 'satellite/S3*res.csv'))
for satfile in satfiles:
    title = '_'.join(satfile.split(r'_')[-3:]).replace('.csv', '')

    print(title)
    df = pd.read_csv(satfile)
    sza = df.SZA.mean()

    solver = Rrs_inversion(wl_sat, a_star_sat, bb_star_sat,sza)
    solver_hyp = Rrs_inversion(wl, a_star, bb_star,sza)

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(21, 21))  # ncols=3, figsize=(30, 12))
    for irow, chl in  enumerate(([0,10],[10,80],[80,1000])):
        norm = mpl.colors.Normalize(vmin=chl[0], vmax=min(200,chl[1]))
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        df_ = df[(df.chl>=chl[0])&(df.chl<chl[1])]
        Rrs_sat = df_.filter(regex='reflectance')
        #Rrs_sat = df_.filter(regex='Rrs_B[0-9]').astype('float')

        # remove SWIR bands:
        Rrs_ = Rrs_sat.iloc[:,:-2].values
        wl_ = [wl_sat] * Rrs_.shape[0]
        x = df_.iloc[:,-5:].values
        #lc = uplot.multiline(wl_, Rrs_, x[:,0], ax=axs[irow,1], cmap='Spectral', lw=2,alpha=0.5)
        xy_max = Rrs_.max() * 1.2 #(max(max(Rrs_sat), max(Rrs_est)) * 1.2)
        for idx, sat in df_.iterrows():
            Rrs_sat = sat.filter(regex='reflectance').astype('float')
            #Rrs_sat = sat.filter(regex='Rrs_B[0-9]').astype('float')
            # remove SWIR bands:
           # Rrs_sat = Rrs_sat.iloc[:-2]
            x = sat.iloc[-5:]

            axs[irow,0].plot(wl_sat, Rrs_sat, color=cmap(norm(x[0])), label='Measured', lw=2.5, alpha=0.75)
            # axs[0].plot(wl_sat, Rrs_est, '--',color=cmap(norm(x[0])), label='Estimation', lw=2.5, alpha=0.75)
            Rrs_est = solver_hyp.forward_model(x)
            # axs[1].plot(wl_sat, Rrs_sat, color=cmap(x[0]), label='Measured', lw=2.5, alpha=0.75)
            axs[irow,1].plot(wl, Rrs_est, '--', color=cmap(norm(x[0])), label='Estimation', lw=2.5, alpha=0.75)
            Rrs_est = solver.forward_model(x)
            compar =axs[irow,2].scatter(Rrs_sat, Rrs_est, c=wl_sat,cmap='Spectral_r', alpha=0.6)

        axs[irow,0].set_ylabel(r'$Rrs\ (sr^{-1})$')
        axs[irow,0].set_xlabel(r'Wavelength (nm)')
        axs[irow,0].set_title(str(chl[0])+r'$ \leq chl <$'+str(chl[1])+', Satellite')
        axs[irow,0].set_xlim(390, 1020)
        axs[irow,1].set_ylabel(r'$Rrs\ (sr^{-1})$')
        axs[irow,1].set_xlabel(r'Wavelength (nm)')
        axs[irow,1].set_title(r'Reconstructed')
        axs[irow,1].set_xlim(390, 1020)
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
    fig.savefig(os.path.join(figdir, 'Rrs_retieval_compar_' + title + '.png'),dpi=200)
    plt.close()
    data = df.iloc[:, -5:]
    data.hist()
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.86])
    #plt.show()


