
import os, sys
import glob
import numpy as np
import netCDF4 as nc
import xarray as xr

# ----------------------------
# set plotting styles
import cmocean as cm
import matplotlib.pyplot as plt

plt.ioff()
plt.rcParams.update({'font.size': 16})

# ----------------------------
# sys.path.extend(['/home/harmel/Dropbox/work/VRTC/OSOAA_profile/exe/lut_package'])
from RTxploitation import lutplot


def walktree(top):
    values = top.groups.values()
    yield values
    for value in top.groups.values():
        for children in walktree(value):
            yield children


idir = '/home/harmel/VRTC/lut/sediment_density'
# idir = '/home/harmel/VRTC/lut/nosea'


odir = '/DATA/projet/VRTC/fig/nosea'
odir = '/DATA/projet/VRTC/fig/bagré'

pattern = 'osoaa_nosea_aot0.1_aero_rg0.80_sig0.60_nr1.51_ni-0.02_ws2_wl*_pressure1015.2.nc'
pattern = 'osoaa_tot_aot0.1_aero_rg0.10_sig0.46_nr1.45_ni-0.0010_ws2_chl3.00_sed200.00_*_wl0.865.nc'

idir = '/sat_data/vrtc/lut/scat_mat'
odir = '/sat_data/vrtc/lut/scat_mat/fig'
pattern = 'osoaa_tot_aot0.1*run036*nc'

idir = '/sat_data/vrtc/lut/sediment_osoaa_v2'
idir = '/sat_data/vrtc/lut/sediment_density'
idir = '/sat_data/vrtc/lut/sediment_junge_v2'

odir = os.path.join(idir,'fig')
pattern = 'osoaa_tot_aot0.1_aero_rg0.80*sed10.0*rmin0.0100_mr1.17_mi-0.001*440.nc'

idir = '/sat_data/vrtc/lut/scat_mat'
odir = '/sat_data/vrtc/lut/scat_mat/fig'
pattern='osoaa_tot_aot0.1*nr1.15_rmed1.0_wl0.400.nc'


files = sorted(glob.glob(os.path.join(idir, pattern)))
directions = ['down', 'up']
direction = directions[1]
lp = lutplot.plot()
vzamax = 61
szaidx = 3

levels = [-1, -2, -3]
Nrow = len(levels)
wl = []

for file in files:

    lut = nc.Dataset(file)
    wl.append(float(lut.getncattr('OSOAA.Wa')) * 1e3)

    figfile = os.path.join(odir, os.path.basename(file).replace('.nc', '') + '_3z')
    # print out group elements
    for children in walktree(lut):
        for child in children:
            print(child)

    sza = lut.variables['sza'][:]

    fig, axs = plt.subplots(Nrow, 5, figsize=(29, 13), subplot_kw=dict(projection='polar'))
    if Nrow == 1:
        axs = np.expand_dims(axs, axis=0)
    fig.subplots_adjust(top=0.9)


    # ----------------------------
    # get data group
    stokes = lut['stokes/' + direction]

    # ----------------------------
    # get dimensions
    z = stokes.variables['z'][:]
    vza = stokes.variables['vza'][:]
    if direction == 'down':
        vza = 180 - vza
    azi = stokes.variables['azi'][:]

    # ----------------------------
    # construct raster dimensions
    r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))

    # ----------------------------
    # get data values
    Idf = stokes.variables['I'][:]
    Qdf = stokes.variables['Q'][:]
    Udf = stokes.variables['U'][:]

    for i, zidx in enumerate(levels):
        # ----------------------------
        # slice and reshape arrays
        I = Idf[zidx, szaidx, vza < vzamax, ...].T
        Q = Qdf[zidx, szaidx, vza < vzamax, ...].T
        U = Udf[zidx, szaidx, vza < vzamax, ...].T
        DOP = (Q ** 2 + U ** 2) ** 0.5 / I

        U_ = xr.DataArray(U, dims=['azi', 'vza'], coords=dict(azi=azi, vza=vza[vza < vzamax]))
        Q_ = xr.DataArray(Q, dims=['azi', 'vza'], coords=dict(azi=azi, vza=vza[vza < vzamax]))
        azi_, vza_ = np.linspace(0, 360, 721), np.linspace(0, vzamax, 101)
        r_, theta_ = np.meshgrid(vza_, np.radians(azi_))
        U_ = U_.interp(azi=azi_,method='cubic').interp(vza=vza_,method='cubic')
        Q_ = Q_.interp(azi=azi_,method='cubic').interp(vza=vza_,method='cubic')

        ratio = -U_/Q_
        #ratio[np.abs(ratio) > 100] = np.nan
        AOP = 1./2 *np.sign(-Q_) *np.abs(np.arctan(ratio))



        # ----------------------------
        # plot polar diagrams
        cmap = cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
        lp.add_polplot(axs[i, 0], r, theta, I, title='I(' + direction + ')', cmap=cmap)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
        lp.add_polplot(axs[i, 1], r, theta, Q, title='Q(' + direction + ')', cmap=cmap)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
        lp.add_polplot(axs[i, 2], r, theta, U, title='U(' + direction + ')', cmap=cmap)
        cmap_dop = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
        lp.add_polplot(axs[i, 3], r, theta, DOP, title='DoLP(' + direction + ')', cmap=cmap_dop, colfmt='%0.2f')

        # cmap = mpl.colors.LinearSegmentedColormap.from_list("",
        #                                             ['navy', "blue", 'lightskyblue',
        #                                              "grey", 'forestgreen', 'yellowgreen',
        #                                              "white", "gold", "darkgoldenrod",
        #                                              'orangered', "firebrick", 'purple'], N=100)
        cmax = int(AOP.__abs__().max()*100)/100
        lp.add_polplot(axs[i, 4], r_, theta_, AOP, title='AoLP(' + direction + ')', cmap=cmap, colfmt='%0.2f',vmin=-cmax,vmax=cmax)

    plt.suptitle(os.path.basename(file) + ' at SZA = ' + str(sza[szaidx]) + ' deg', fontdict=lp.font)
    plt.tight_layout()
    plt.savefig(figfile +'_sza' + str(sza[szaidx]) +'.png', dpi=300, bbox_inches='tight')
    plt.close()
