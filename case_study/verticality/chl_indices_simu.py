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

from RTxploitation import lutplot
from RTxploitation import utils as u


def walktree(top):
    values = top.groups.values()
    yield values
    for value in top.groups.values():
        for children in walktree(value):
            yield children


noclobber = True
project = ''
project = 'chl_profile'
dir_ = ['/DATA/projet/VRTC/', '/home/harmel/VRTC']
idir = os.path.join(dir_[1], 'lut', project)
odir = os.path.join(dir_[1], 'lut', project, 'fig')
wlcs = ['440', '550', '670', '705', '750']
lp = lutplot.plot()
vzamax = 61
szaidx = 3
zidx = -2
level = ''
if zidx == -1:
    level = 'TOA'
    directions = ['up']

wl = []
direction = 'up'

profiles = ('homogeneous', '$z_{max}=2$', '$z_{max}=10$', '$z_{max}=15$')
Nrow = len(profiles)
chl_max = u.misc.arr_format([0, 15, 15, 15], "{:0.2f}")
B0 = u.misc.arr_format([6.7, 2.6, 1.4, 0.4], "{:0.2f}")
z_max = u.misc.arr_format([0, 2, 5, 15], "{:0.1f}")
sig_chl = u.misc.arr_format([0, 5, 5, 5], "{:0.1f}")

Nrow = 1
fig, axs = plt.subplots(Nrow, 4, figsize=(24, Nrow * 6), subplot_kw=dict(projection='polar'))


def get_pattern(wlc, i):
    return 'osoaa_tot_aot0.1_aero_rg0.80_sig0.60_nr1.45_ni-0.0010_ws2guassian_profile_B0' + B0[i] + \
           '_zmax' + z_max[i] + '_sig' + sig_chl[i] + '_chl' + chl_max[i] + \
           '_sed0.00_sedjs3.70_sednr1.17_sedni-0.0001_wl0.' + wlc + '.nc'


def read_lut(wlc, i):
    pattern = get_pattern(wlc, i)
    file = os.path.join(idir, pattern)
    lut = nc.Dataset(file)
    # ----------------------------
    # get data group
    stokes = lut['stokes/' + direction]
    # ----------------------------
    # get data values
    Idf = stokes.variables['I'][:]
    return Idf[zidx, szaidx, vza < vzamax, ...].T


def get_dim_lut(wlc='440'):
    i = 0
    pattern = get_pattern(wlc, i)
    file = os.path.join(idir, pattern)
    lut = nc.Dataset(file)

    sza = lut.variables['sza'][:]

    # ----------------------------
    # get data group
    stokes = lut['stokes/' + direction]
    z = stokes.variables['z'][:]
    vza = stokes.variables['vza'][:]
    azi = stokes.variables['azi'][:]

    return sza, vza, azi, z


sza, vza, azi, z = get_dim_lut()
# ----------------------------
# construct raster dimensions
r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))
szac = str(sza[szaidx])
figfile = os.path.join(odir, 'ratio_blue_green_for_various_chl_profiles_sza' + szac)

for i, profile in enumerate(profiles):
    wlc = '440'

    I440 = read_lut('440', i)
    I550 = read_lut('550', i)

    ratio = I440 / I550

    # ----------------------------
    # plot polar diagrams
    cmap = cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
    lp.add_polplot(axs[i], r, theta, ratio, title='Rrs(440)/Rrs(550); ' + profile, cmap=cmap)

plt.suptitle(os.path.basename(get_pattern(wlc,i)) + ' at SZA = ' + str(sza[szaidx]) + ' deg', fontdict=lp.font)
plt.tight_layout(rect=[0.0, 0.0, 0.99, 0.95])
plt.savefig(figfile + '.png', dpi=300, bbox_inches='tight')
plt.close()
