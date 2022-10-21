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

opj = os.path.join


def walktree(top):
    values = top.groups.values()
    yield values
    for value in top.groups.values():
        for children in walktree(value):
            yield children


idir = '/sat_data/vrtc/lut/atmo'
odir = '/DATA/git/vrtc/RTxploitation/study_cases/aerosol/fig'
pattern = "fine_rm0.10_sig0.40_nr1.48_ni-0.0035_coarse_rm0.81_sig0.60_nr1.48_ni-0.0035_HR8.0_HA2.0_ws2_wl0.550"

directions = ['down', 'up']

lp = lutplot.plot()
vzamax = 61
szaidx = 6
zidx = -1
if zidx == -1:
    directions = ['up']
Nrow = 4#len(directions)

figfile = os.path.join(odir, pattern )
fig, axs = plt.subplots(Nrow, 6, figsize=(30, 4+Nrow*4), subplot_kw=dict(projection='polar'))
fig.subplots_adjust(top=0.9)
if Nrow == 1:
    axs = np.expand_dims(axs, axis=0)
wl = []

CVfs = [0.,0.2,.4,0.6, 0.8, 1]
for iCVf, CVf in enumerate(CVfs):
    file = "osoaa_atmo_aot0.2_CVfine{:.2f}_".format(CVf) + pattern + '.nc'
    pfile = opj(idir, file)


    lut = nc.Dataset(pfile)
    wl.append(float(lut.getncattr('OSOAA.Wa')) * 1e3)

    # print out group elements
    for children in walktree(lut):
        for child in children:
            print(child)

    sza = lut.variables['sza'][:]

    for i, direction in enumerate(directions):
        print(i, direction)

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
        cmap = plt.cm.Spectral_r #cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
        lp.add_polplot(axs[0, iCVf], r, theta, I, title='CVf={:.2f}'.format(CVf), cmap=cmap,minmax=(0.035,0.09))
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
        lp.add_polplot(axs[1, iCVf], r, theta, Q, title='Q(' + direction + ')', cmap=cmap)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
        lp.add_polplot(axs[2, iCVf], r, theta, U, title='U(' + direction + ')', cmap=cmap)
        cmap = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
        lp.add_polplot(axs[3, iCVf], r, theta, DOP, title='DOP(' + direction + ')', cmap=cmap, colfmt='%0.2f')

plt.suptitle(os.path.basename(file) + ' at SZA = ' + str(sza[szaidx]) + ' deg', fontdict=lp.font)
plt.tight_layout()
plt.savefig(figfile +'_sza'+ str(sza[szaidx])+ '.png', dpi=300, bbox_inches='tight')
plt.close()
