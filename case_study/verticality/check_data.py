import os
import numpy as np
from scipy import special
import pandas as pd
import matplotlib.pyplot as plt
import cmocean
import lmfit as lm

opj = os.path.join

optimization=True

idir = os.path.abspath('/DATA/git/vrtc/RTxploitation/study_cases/verticality')
figdir = opj(idir, 'figure')
file = opj(idir, 'data/leman_chlorophylle_1430213014157-121850.csv')

df = pd.read_csv(file, sep=';', index_col=3, parse_dates=True).dropna(axis=1)
df.columns = ['ID', 'site', 'platform', 'depth_min', 'depth_max', 'chl_a']
df['month'] = df.index.month
df['doy'] = df.index.dayofyear

depth = np.arange(0, 30, 0.1)


def simple_gauss_profile(z, B0, chl_gauss, z_max, sig):
    '''
    Chlorophyll profile (in desired unit) simplified from
    Matsumura and Shiomoto’s (1993) equation (Fig. 2) (gradienbt of background neglected)
    :param z: depth in m
    :param B0: [chl] of background (i.e., constant over depths)
    :param chl_gauss: norm of the gaussian [chl] distrib (i.e., chl(z_max) = B0+chl_max/(sig*(2pi)**0.5))
    :param z_max: depth of the chl maximum (mean of the normal distribution (in m))
    :param sig: width/sigma of the normal distribution
    :return:
    '''
    sqrt_2pi = np.sqrt(2 * np.pi)
    chl = B0 + chl_gauss / (sig * sqrt_2pi) * np.exp(-(z - z_max) ** 2 / (2 * sig ** 2))
    return chl


def osoaa_gauss_profile(z, B0, chl_max, z_max, sig):
    '''
    From OSOAA code
    :param z: depth in m
    :param B0: [chl] of background (i.e., constant over depths)
    :param chl_max: chl at z_max above background (i.e., chl(z_max) = B0+chl_max)
    :param z_max: depth of the chl maximum (mean of the normal distribution (in m))
    :param sig: width/sigma of the normal distribution
    :return:
    '''
    chl = B0 + chl_max * np.exp(-(z - z_max) ** 2 / (2 * sig ** 2))
    return chl


def osoaa_gauss_integral(z_down, B0, chl_max, z_max, sig, z_up=0):
    '''
    Analytic integration of the gauss distribution from 'osoaa_gauss_profile'
    :param z_up: start of integration (levby default at the surface z_up=0)
    :param z_down: end of the integration, positive depth in m
    :param B0: [chl] of background (i.e., constant over depths)
    :param chl_max: chl at z_max above background (i.e., chl(z_max) = B0+chl_max)
    :param z_max: depth of the chl maximum (mean of the normal distribution (in m))
    :param sig: width/sigma of the normal distribution
    :return: chl_tot (mg m-2)
    '''
    sqrt_2pi = np.sqrt(2 * np.pi)
    sqrt_2 = 2 ** 0.5
    chl_tot = B0 * (z_down - z_up) + 0.5 * chl_max * sig * sqrt_2pi * \
              (special.erf((z_down - z_max) / (sig * sqrt_2)) -
               special.erf((z_up - z_max) / (sig * sqrt_2)))
    return chl_tot


def return_B0(chl_tot, z_down, chl_max, z_max, sig, z_up=0):
    '''
    Return value of B0 for a given Chl_tot value and prescribed profile parameters
    :param chl_tot: integrated chl_value in mg.m-2
    :param z_up: start of integration (levby default at the surface z_up=0)
    :param z_down: end of the integration, positive depth in m

    :param chl_max: chl at z_max above background (i.e., chl(z_max) = B0+chl_max)
    :param z_max: depth of the chl maximum (mean of the normal distribution (in m))
    :param sig: width/sigma of the normal distribution
    :return: B0 (mg m-3)
    '''
    sqrt_2pi = np.sqrt(2 * np.pi)
    sqrt_2 = 2 ** 0.5
    B0 = chl_tot - 0.5 * chl_max * sig * sqrt_2pi * \
         (special.erf((z_down - z_max) / (sig * sqrt_2)) -
          special.erf((z_up - z_max) / (sig * sqrt_2)))
    B0 = B0 / (z_down - z_up)
    if B0 < 0:
        print('Chl_tot and paramaters provided lead to impossible negative B0')
        print('please check it again')
        print('for now B0 is set to 0')
        B0=0
    return B0


def objfunc(x, z, chl):
    B0, chl_max, z_max, sig = np.array(list(x.valuesdict().values()))
    chl_est = osoaa_gauss_profile(z, B0, chl_max, z_max, sig)
    return chl - chl_est


B0_max = df[df.depth_min == 30].chl_a.max()


def solver(z, chl):
    pars = lm.Parameters()
    pars.add('B0', value=0.05, min=0, max=B0_max)
    pars.add('chl_max', value=1, min=0, max=200)
    pars.add('z_max', value=5, min=-5, max=21)
    pars.add('sig', value=5, min=2.)

    min1 = lm.Minimizer(objfunc, pars, fcn_args=(z, chl))

    out1 = min1.least_squares(max_nfev=30, xtol=1e-7, ftol=1e-4)
    out1.params.pretty_print()
    return out1


# by trimester
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()
trimestre = ['JFM', 'AMJ', 'JAS', 'OND']
for idx, period in enumerate(([1, 3], [4, 6], [7, 9], [10, 12])):
    print(idx, period[0])

    df[(df.month >= period[0]) & (df.month <= period[1])].plot.scatter('chl_a', 'depth_min', c='doy', cmap='Spectral',
                                                                       ax=axs[idx])

    axs[idx].set_title(trimestre[idx])
    axs[idx].invert_yaxis()

if optimization:
    # by month
    plt.ioff()
    reslist = []
    dfy = df  # [df.index.year==2008]
    imonths =range(12)
    #imonths = [0,2,4,7]
    nrow = int(len(imonths)/4)
    fig, axs = plt.subplots(nrow, 4, figsize=(13, 5.2*nrow))
    axs = axs.ravel()
    # df.index.strftime('%B').unique()
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
    cmap = plt.get_cmap('Spectral_r')

    for i,imonth in enumerate(imonths):
        dfm = dfy[dfy.month == imonth + 1]
        ii = 0
        data = dfm.groupby(level=0)
        N = data.ngroups
        for g_, df_ in data:
            # -------------
            # optimization
            z, chl = df_.depth_min, df_['chl_a'].values
            res = solver(z, chl)
            reslist.append([g_, *res.x])
            # -------------

            color = cmap(float(ii) / N)
            axs[i].plot(osoaa_gauss_profile(depth, *res.x), depth, '--', c=color, lw=2.5, alpha=0.6)
            axs[i].plot(df_['chl_a'], df_.depth_min, '-o', c=color, alpha=0.6)
            ii += 1

        axs[i].invert_yaxis()
        axs[i].set_ylabel('Depth (m)')
        axs[i].set_xlabel('Chl-a (mg $m^{-3})$')
        axs[i].set_title(months[imonth])

    resdf = pd.DataFrame.from_records(reslist)
    resdf.columns = ['date', 'B0', 'chl_max', 'z_max', 'sig']
    resdf['date'] = pd.to_datetime(resdf.date)
    resdf.to_csv(opj(figdir, 'gaussian_fit_chl_leman.csv'), index=False)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.subplots_adjust(left=0.1, right=0.9, hspace=.3, wspace=0.45)
    plt.suptitle('Lake Leman, Chl from SHL2')
    fig.savefig(opj(figdir, 'gaussian_fit_chl_Leman.png'), dpi=200)
    fig.savefig(opj(figdir, 'gaussian_fit_chl_Leman.pdf'))

# histogram
resdf = pd.read_csv(opj(figdir, 'gaussian_fit_chl_leman.csv'),index_col=0,parse_dates=True)
resdf['month'] = resdf.index.month
resdf['year_quarter'] = resdf.index.quarter
pardate = 'year_quarter'
# compute integrated value Chl_tot
resdf['chl_tot']=resdf.apply(lambda row: osoaa_gauss_integral(30,row['B0'], row['chl_max'],row['z_max'],row['sig']), axis=1)

import seaborn as sns

sns.set_style('white')
g = sns.pairplot(resdf.loc[:, (pardate, 'chl_tot', 'chl_max', 'z_max', 'sig')], hue=pardate, palette='Spectral', aspect=1.5)

g.savefig(opj(figdir, 'histogram_gaussian_fit_chl_Leman.pdf'))
g.savefig(opj(figdir, 'histogram_gaussian_fit_chl_Leman.png'), dpi=200)

# fig for highly peaked profile
chl_threshold = 10
resdf_ = resdf[resdf.chl_max > chl_threshold]
g = sns.pairplot(resdf_.loc[:, ('B0', 'chl_max', 'z_max', 'sig')], palette='Spectral', aspect=1.5)

# fig for simulation cases
plt.ioff()
import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(6, 9))
chl_tot=200
chl_max=15
sig=5
for z_max in [2,5,15]:
    B0=return_B0(chl_tot,30,chl_max, z_max, sig)
    label='$B_0=%.1f,Chl_{max}=%2d,z_{max}=%2d,\sigma=%2d$'%(B0,chl_max, z_max, sig)
    ax.plot(osoaa_gauss_profile(depth, B0, chl_max, z_max, sig), depth, '--', label=label, lw=2.5, alpha=0.6)
chl_max=0
B0=return_B0(chl_tot,30,chl_max, z_max, sig)
label='$B_0=%.1f,Chl_{max}=%2d}$'%(B0,chl_max, )
ax.plot(osoaa_gauss_profile(depth, B0, chl_max, z_max, sig), depth, 'k--', label=label, lw=2.5)

ax.legend(prop={'size': 12})
ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Chl-a (mg $m^{-3})$')
ax.grid()
fig.savefig(opj(figdir, 'tabulated_profile_for_RT.png'), dpi=200)