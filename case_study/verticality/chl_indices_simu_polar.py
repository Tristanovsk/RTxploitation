
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
project=''
project='chl_profile'
dir_ = ['/DATA/projet/VRTC/','/home/harmel/VRTC']
idir = os.path.join(dir_[1],'lut',project)
odir = os.path.join(dir_[1],'lut',project,'fig')
wlcs=['440','550','670','705','750']
lp = lutplot.plot()
vzamax = 61
szaidx = 3
zidx=-2
level=''
if zidx == -1:
    level='TOA'
    directions = [ 'up']

wl=[]
direction='up'

profiles=('homogeneous','$z_{max}=2$','$z_{max}=10$','$z_{max}=15$')
Nrow=len(profiles)
chl_max = u.misc.arr_format([0,15,15,15],"{:0.2f}")
B0=u.misc.arr_format([ 6.7,2.6,1.4,0.4],"{:0.2f}")
z_max=u.misc.arr_format([0, 2,5,15],"{:0.1f}")
sig_chl=u.misc.arr_format([0, 5,5,5],"{:0.1f}")

for wlc in  wlcs:



    fig, axs = plt.subplots(Nrow, 4, figsize=(24, Nrow*6), subplot_kw=dict(projection='polar'))




    for i, profile in enumerate(profiles):

        pattern='osoaa_tot_aot0.1_aero_rg0.80_sig0.60_nr1.45_ni-0.0010_ws2guassian_profile_B0'+B0[i]+\
                '_zmax'+z_max[i]+'_sig'+sig_chl[i]+'_chl'+chl_max[i]+\
                '_sed0.00_sedjs3.70_sednr1.17_sedni-0.0001_wl0.'+wlc+'.nc'
        file = os.path.join(idir, pattern)
        lut = nc.Dataset(file)
        wl.append(float(lut.getncattr('OSOAA.Wa'))*1e3)

        sza = lut.variables['sza'][:]
        if i == 0:
            szac = str(sza[szaidx])
            figfile = os.path.join(odir, 'radiance_for_various_chl_profiles_sza' + szac + '_wl' + wlc)

            #axs = np.expand_dims(axs, axis=0)

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
        Idf=stokes.variables['I'][:]
        Qdf = stokes.variables['Q'][:]
        Udf = stokes.variables['U'][:]

        # ----------------------------
        # slice and reshape arrays
        I = Idf[zidx, szaidx, vza < vzamax, ...].T
        Q = Qdf[zidx, szaidx, vza < vzamax, ...].T
        U = Udf[zidx, szaidx, vza < vzamax, ...].T
        DOP = (Q ** 2 + U ** 2) ** 0.5 / I
        if i == 0:
            minmaxI=[np.min(I)*1.1,np.max(I)*0.9]
            minmaxQ=[np.min(Q)*1.1,np.max(Q)*0.9]
            minmaxU=[np.min(U)*1.1,np.max(U)*0.9]
            minmaxDOP=[0,np.max(DOP)]
            if wlc == '550':
                minmaxI=[5e-3,3e-2]
                minmaxU=[-4e-3,4e-3]
                minmaxDOP=[0,0.7]
        # ----------------------------
        # plot polar diagrams
        cmap = cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
        lp.add_polplot(axs[i, 0], r, theta, I, title='I(profile ' + profile + ')', minmax=minmaxI, cmap=cmap)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
        lp.add_polplot(axs[i, 1], r, theta, Q, title='Q(profile ' + profile + ')', minmax=minmaxQ, cmap=cmap)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
        lp.add_polplot(axs[i, 2], r, theta, U, title='U(profile ' + profile + ')', minmax=minmaxU, cmap=cmap)
        cmap = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
        lp.add_polplot(axs[i, 3], r, theta, DOP, title='DOP(profile ' + profile + ')', minmax=minmaxDOP, cmap=cmap, colfmt='%0.2f')

    plt.suptitle(os.path.basename(file) + ' at SZA = ' + str(sza[szaidx]) + ' deg', fontdict=lp.font)
    plt.tight_layout(rect=[0.0, 0.0, 0.99, 0.95])
    plt.savefig(figfile + '.png', dpi=300, bbox_inches='tight')
    plt.close()
