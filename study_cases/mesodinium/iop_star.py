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

plt.ioff()
plt.rcParams.update({'font.size': 18})

opj = os.path.join
idir = '/DATA/projet/gernez/mesodinium'
datadir = opj(idir, 'data')
figdir = opj(idir, 'fig')

# -----------------
# check bb and a star
# -----------------
hplc = pd.read_excel('/DATA/projet/gernez/data/HPLC_data.xlsx',index_col=3)
hplc.index = pd.MultiIndex.from_tuples(list(hplc.index.str.split('_')),names=['strain','num'])
hplc.columns= pd.MultiIndex.from_product([hplc.columns,['']])

bbp = pd.read_csv('/DATA/projet/gernez/data/bbp/bbp_all.txt',index_col=0,header=[0,1])
bbp.index = pd.MultiIndex.from_tuples(list(bbp.index.str.split('_')),names=['strain','num'])

iop_star=hplc.join(bbp)
plt.ioff()
df =iop_star.loc['CIL']
x = df.loc[:,'Chlorophyll a']
bbp = df.loc[:,'bbp']
bbp_SD = df.loc[:,'bbp_SD']

def confidence_interval(xn, model, res, nstd=1):
    '''

    :param xn: x-axis data for computation
    :param model: numerical model used for the fit
    :param res: output from scipy.odr.run
    :param nstd: number of sigma to compute confidence interval
    :return: data up and data down
    '''
    '''
    
    :param res: output from scipy.odr.run
    :param nstd: number of sigma to compute confidence interval
    :return: 
    '''

    popt_up = res.beta + nstd * res.sd_beta
    popt_dw = res.beta - nstd * res.sd_beta

    return model(popt_up, xn), model(popt_dw, xn)

def linear(B, x):
    '''Linear function y = m*x + b'''
    return B[0] + B[1] * x

def linear0(B, x):
    '''Linear function y = m*x + b'''
    return  B[0] * x

model, N = linear, 2
model, N = linear0, 1


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
axs=axs.ravel()

for i, wl in enumerate(bbp.columns):
    y = bbp.values[:,i]
    yerr = bbp_SD.values[:,i]
    xerr=0.1*x

    axs[i].errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color='grey', ecolor='black', alpha=0.8)
    axs[i].set_title('wl = '+wl+' nm')
    axs[i].set_ylabel(r'$b_{bp}\ (m^{-1})$')
    axs[i].set_xlabel(r'Chl-a $(mg\ m^{-3})$')

    testdata = odr.RealData(x, y, sx=xerr, sy=yerr)
    _odr = odr.ODR(testdata, odr.Model(model), beta0=[1.] * N)
    _odr.set_job(fit_type=0)
    res = _odr.run()
    res.pprint()
    xn = np.linspace(0, np.max(x) * 1.25, 50)
    yn = model(res.beta, xn)
    fit_up, fit_dw = confidence_interval(xn, model, res)
    if N == 1:
        a = res.beta[0]
        axs[i].plot(xn, yn, 'r-',
                    label='y = {:.2e}x\n $\Delta slope$ = {:.1e}'.format(a, res.sd_beta[N - 1]),
                    linewidth=2)
    else:
        a, b = res.beta[1], res.beta[0]
        axs[i].plot(xn, yn, 'r-',
                    label='y = {:.2e}x + {:.1e}\n $\Delta slope$ = {:.1e}'.format(a, b, res.sd_beta[N - 1]),
                    linewidth=2)
    axs[i].fill_between(xn, fit_up, fit_dw, alpha=.25, facecolor="r")  # ,label="1-sigma interval")
    axs[i].legend()
plt.suptitle('Mesodinium Rubrum')
plt.tight_layout()
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.89])
fig.savefig(opj(figdir,'bbp_star_calib_0origin.png'), dpi=300)