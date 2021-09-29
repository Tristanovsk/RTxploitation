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
idir = '/sat_data/vrtc/lut/scat_mat'

odir = '/DATA/projet/VRTC/fig/nosea'
odir = '/DATA/projet/VRTC/fig/bagré'
odir = '/sat_data/vrtc/lut/scat_mat/fig'
pattern = 'osoaa_nosea_aot0.1_aero_rg0.80_sig0.60_nr1.51_ni-0.02_ws2_wl*_pressure1015.2.nc'
pattern = 'osoaa_tot_aot0.1_aero_rg0.10_sig0.46_nr1.45_ni-0.0010_ws2_chl3.00_sed200.00_*_wl0.865.nc'

pattern = 'osoaa_tot_aot0.1*run038*nc'

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

    fig, axs = plt.subplots(Nrow, 4, figsize=(24, 13), subplot_kw=dict(projection='polar'))
    if Nrow == 1:
        axs = np.expand_dims(axs, axis=0)
    fig.subplots_adjust(top=0.9)
    for i, zidx in enumerate(levels):

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

        # ----------------------------
        # slice and reshape arrays
        I = Idf[zidx, szaidx, vza < vzamax, ...].T
        Q = Qdf[zidx, szaidx, vza < vzamax, ...].T
        U = Udf[zidx, szaidx, vza < vzamax, ...].T
        DOP = (Q ** 2 + U ** 2) ** 0.5 / I

        # ----------------------------
        # plot polar diagrams
        cmap = cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
        lp.add_polplot(axs[i, 0], r, theta, I, title='I(' + direction + ')', cmap=cmap)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
        lp.add_polplot(axs[i, 1], r, theta, Q, title='Q(' + direction + ')', cmap=cmap)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
        lp.add_polplot(axs[i, 2], r, theta, U, title='U(' + direction + ')', cmap=cmap)
        cmap = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
        lp.add_polplot(axs[i, 3], r, theta, DOP, title='DOP(' + direction + ')', cmap=cmap, colfmt='%0.2f')

    plt.suptitle(os.path.basename(file) + ' at SZA = ' + str(sza[szaidx]) + ' deg', fontdict=lp.font)
    plt.tight_layout()
    plt.savefig(figfile + '.png', dpi=300, bbox_inches='tight')
    plt.close()
