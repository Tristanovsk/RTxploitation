import os, sys
import glob
import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd

# ----------------------------
# set plotting styles
import cmocean as cm
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

# TODO check sunglint component from OSOAA check OSOAA_SOS_CORE_harmel.F line 2181 or 3982

# ----------------------------

# sys.path.extend(['/home/harmel/Dropbox/work/VRTC/OSOAA_profile/RTxploitation'])
from RTxploitation import lutplot
from RTxploitation import utils as RTu

models = ['rg0.10_sig0.46_nr1.51_ni-0.0200', 'rg0.10_sig0.46_nr1.45_ni-0.0001',
          'rg0.80_sig0.60_nr1.51_ni-0.0200', 'rg0.80_sig0.60_nr1.35_ni-0.0010']
model = models[2]
project = 'nosea'
idir = os.path.join('/DATA/projet/VRTC/lut/', project)
odir = os.path.join('/DATA/projet/VRTC/fig/', project)
idir = '/home/harmel/VRTC/lut/nosea/'

aot_list = ('0.01', '0.1', '0.2', '0.5', '1.0')
aot_num = [float(x) for x in aot_list]
wl_num = [0.400, 0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.865, 1.610, 2.190, 2.4]
wl_list = ["{:0.3f}".format(x) for x in wl_num]

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


def to_netcdf(idir, model):
    first = True
    aot_ = np.empty((len(aot_list), len(wl_list)))
    ofile = os.path.join(odir, 'lut_toa_rad_aero_' + model + '.nc')

    aerosol_props = pd.DataFrame()
    for iaot, aot in enumerate(aot_list):
        for iwl, wl in enumerate(wl_list):

            pattern = 'osoaa_nosea_aot' + aot + '_aero_' + model + '_ws2_wl' + wl + '_pressure1015.2.nc'

            file = os.path.join(idir, pattern)
            print(iaot, iwl, file)
            lut = nc.Dataset(file)
            if iaot == 0:
                aerosol = lut['/optical_properties/aerosol'].variables
                keys = list(aerosol.keys())
                ds = pd.DataFrame(index={iwl})
                for key in ['wl', 'wl_ref', 'aot_ref', 'Cext', 'Cext_ref', 'ssa_ref', 'ssa', 'asym_factor',
                            'vol_mean_particle', 'nr', 'ni', 'reff', 'vareff']:  # keys:
                    ds[key] = aerosol[key][:]

                aerosol_props = pd.concat([aerosol_props, ds])

            # if you want to check netcdf content (e.g., groups):
            RTu.nc_utils().print_groups(lut)

            wl_.append(float(lut.getncattr('OSOAA.Wa')) * 1e3)
            sza = lut.variables['sza'][:]
            aot_[iaot, iwl] = lut.groups['optical_properties'].groups['aerosol'].variables['aot'][:]

            for i, direction in enumerate(directions):
                print(i, direction)

                # ----------------------------
                # get data group
                stokes = lut['stokes/' + direction]

                # ----------------------------
                # initialize arrays within first loop
                if first:
                    Nlut = stokes.variables['I'][:].shape
                    dim = (len(aot_list), len(wl_list)) + Nlut
                    Iu, Qu, Uu, Id, Qd, Ud = init_array(6, dim)
                    first = False

                # ----------------------------
                # get dimensions
                z = stokes.variables['z'][:]
                vza = stokes.variables['vza'][:]
                azi = stokes.variables['azi'][:]

                # ----------------------------
                # get data values
                if direction == 'down':
                    vza = 180 - vza
                    Id[iaot, iwl, ...] = stokes.variables['I'][:]
                    Qd[iaot, iwl, ...] = stokes.variables['Q'][:]
                    Ud[iaot, iwl, ...] = stokes.variables['U'][:]
                elif direction == 'up':
                    Iu[iaot, iwl, ...] = stokes.variables['I'][:]
                    Qu[iaot, iwl, ...] = stokes.variables['Q'][:]
                    Uu[iaot, iwl, ...] = stokes.variables['U'][:]

    aerosol_props = aerosol_props.set_index('wl').to_xarray()

    def _toxr(arr):
        arr = np.array(arr)
        return xr.DataArray(arr,
                            dims=('aot', 'wl', 'z', 'sza', 'vza', 'azi'),
                            coords={'aot': aot_num, 'wl': wl_num, 'z': z, 'sza': sza, 'vza': vza, 'azi': azi})

    Idxr, Qdxr, Udxr = _toxr(Id), _toxr(Qd), _toxr(Ud)
    Iuxr, Quxr, Uuxr = _toxr(Iu), _toxr(Qu), _toxr(Uu)

    aotxr = xr.DataArray(aot_, dims=('aot', 'wl'), coords={'aot': aot_num, 'wl': wl_num})
    if os.path.exists(ofile):
        os.remove(ofile)
    Iuxr.to_netcdf(ofile, group='Lnorm', mode='w')
    aerosol_props.to_netcdf(ofile, group='aerosol', mode='a')


for model in models:
    to_netcdf(idir, model)
