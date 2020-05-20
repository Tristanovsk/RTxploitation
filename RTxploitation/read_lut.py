import os, sys
import glob
import numpy as np
import netCDF4 as nc
import xarray as xr

# ----------------------------
# set plotting styles
import cmocean as cm
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

# TODO check sunglint component from OSOAA check OSOAA_SOS_CORE_harmel.F line 2181 or 3982

from RTxploitation import lutplot
from RTxploitation import utils as u

idir = '/home/harmel/VRTC/lut/'
odir = '/home/harmel/VRTC/fig/'
pattern = 'osoaa_nosea_aot0.1_aero_rg0.80_sig0.60_nr1.51_ni-0.02_ws2_wl*_pressure1015.2.nc'
pattern = 'osoaa_nosea_aot0.1_aero_rg0.10_sig0.46_nr1.51_ni-0.02_ws2_wl*_pressure1015.2.nc'
files = sorted(glob.glob(os.path.join(idir, pattern)))

aot_list = ('0.01', '0.1', '0.2', '0.5', '1.0')
aot_num = [ float(x) for x in aot_list ]
wl_num=[0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.865, 1.610, 2.190]
wl_list = [ "{:0.3f}".format(x) for x in wl_num ]


directions = ['down', 'up']

lp = lutplot.plot()
vzamax = 61
szaidx = 3
zidx = 1

Nrow = len(directions)
wl_ = []


def init_array(number_of_array, dim):
    '''
    Initialize Nd array, Nd dimensions are given by dim
    example:
        arr1, arr2 = init_array(2,(2,3,10))
    gives
        arr1.shape --> (2,3,10)
        arr2.shape --> (2,3,10)

    :param number_of_array: int
    :param dim: list
    :return: `number_of_array` numpy arrays of shape `dim`
    '''
    for i in range(number_of_array):
        yield np.empty(dim)

first=True
for iaot, aot in enumerate(aot_list):
    for iwl, wl in enumerate(wl_list):
        
        pattern = 'osoaa_nosea_aot'+aot+'_aero_rg0.10_sig0.46_nr1.51_ni-0.0200_ws2_wl'+wl+'_pressure1015.2.nc'
        print(iaot,iwl,pattern)
        file = os.path.join(idir, pattern)
        figfile = os.path.join(odir, os.path.basename(file).replace('.nc', ''))

        lut = nc.Dataset(file)

        # if you want to check netcdf content (e.g., groups):
        u.nc_utils().print_groups(lut)

        wl_.append(float(lut.getncattr('OSOAA.Wa')) * 1e3)
        sza = lut.variables['sza'][:]



        for i, direction in enumerate(directions):
            print(i, direction)

            # ----------------------------
            # get data group
            stokes = lut['stokes/' + direction]

            # ----------------------------
            # initialize arrays within first loop
            if first:
                Nlut = stokes.variables['I'][:].shape
                dim = (len(aot_list),len(wl_list))+Nlut
                Iu, Qu, Uu, Id, Qd, Ud = init_array(6,dim)
                first=False

            # ----------------------------
            # get dimensions
            z = stokes.variables['z'][:]
            vza = stokes.variables['vza'][:]
            azi = stokes.variables['azi'][:]

            # ----------------------------
            # get data values
            if direction == 'down':
                vza = 180 - vza
                Id[iaot,iwl,...]=stokes.variables['I'][:]
                Qd[iaot,iwl,...]=stokes.variables['Q'][:]
                Ud[iaot,iwl,...]=stokes.variables['U'][:]
            elif direction == 'up':
                Iu[iaot,iwl,...]=stokes.variables['I'][:]
                Qu[iaot,iwl,...]=stokes.variables['Q'][:]
                Uu[iaot,iwl,...]=stokes.variables['U'][:]
               
def _toxr(arr):
    arr = np.array(arr)
    return xr.DataArray(arr,
                 dims=('aot','wl', 'z', 'sza', 'vza', 'azi'),
                 coords={'aot':aot_num,'wl': wl_num, 'z': z, 'sza': sza, 'vza': vza, 'azi': azi})

Idxr, Qdxr, Udxr = _toxr(Id), _toxr(Qd), _toxr(Ud)
Iuxr, Quxr, Uuxr = _toxr(Iu), _toxr(Qu), _toxr(Uu)

# test plots
zidx=0
vzaidx=27
fig, axs = plt.subplots(1,2, figsize=(15, 6))
Idxr.isel(z=zidx, sza=2, azi=18, vza=vzaidx).plot.line(ax=axs[0],x='wl',marker='o')
Iuxr.isel(z=zidx, sza=2, azi=18, vza=vzaidx).plot.line(ax=axs[1],x='wl',marker='o')

rho_factor = Iuxr.isel(z=0) / Idxr.isel(z=0)
fig, axs = plt.subplots(1,2, figsize=(15, 6))
rho_factor.isel( sza=2, azi=18, vza=vzaidx).plot.line(ax=axs[0],x='wl',marker='o')
rho_factor.isel( sza=6, azi=18, vza=vzaidx).plot.line(ax=axs[1],x='wl',marker='o')

# ----------------------------

# construct raster dimensions
r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))

fig, axs = plt.subplots(Nrow, 4, figsize=(24, 13), subplot_kw=dict(projection='polar'))
if Nrow == 1:
    axs = np.expand_dims(axs, axis=0)
fig.subplots_adjust(top=0.9)

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
