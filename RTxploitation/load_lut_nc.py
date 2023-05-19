import os, sys

opj = os.path.join
import glob
import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd

# ----------------------------
# set plotting styles
import cmocean as cm
import matplotlib.pyplot as plt
import seaborn as sns

plt.ioff()
plt.rcParams.update({'font.size': 16})


def s2f(list):
    return np.array(list).astype(np.float)


def load_osoaa(prefix='osoaa_tot_aot0.1_aero_rg0.10_sig0.46_nr1.45_ni-0.0010_ws2',
               labels=['__sed10.00_nr1.2_rmed10.0_wl0.400', '__sed10.00_nr1.2_rmed10.0_wl0.500'],
               direction='up',
               idir='/sat_data/vrtc/lut/scat_mat/',
               water_signal=False,
               z_slice=[-3, -2, -1],
               ):
    '''

    :param prefix:
    :param labels:
    :param direction:
    :param idir:
    :param water_signal:
    :param IQU:
    :return:
    '''

    xStokes = []
    for label in labels:
        file_ = prefix + label + '.nc'
        file = os.path.join(idir, file_)
        print(file)

        if not os.path.exists(file):
            print('Warning %s does not exist' % file)
            continue
        lut = nc.Dataset(file)
        wl = lut.getncattr('OSOAA.Wa')
        sza = lut.variables['sza'][:]
        Stokes = xr.open_dataset(file, group='stokes/' + direction).assign_coords({'sza': sza}).isel(z=z_slice)
        flux = xr.open_dataset(file, group='flux').assign_coords({'sza': sza}).sel(z=Stokes.z)
        if water_signal:
            file_nosea = os.path.join(idir,prefix.replace('_tot', '_nosea') + '_wl%s.nc' % wl)
            lut_nosea = nc.Dataset(file_nosea)
            sza_ = lut.variables['sza'][:]
            Snosea = xr.open_dataset(file_nosea, group='stokes/' + direction).assign_coords({'sza': sza_})
            Stokes = Stokes - Snosea

        sza = Stokes.sza
        Rstokes = Stokes / np.cos(np.radians(sza)) / np.pi
        xStokes_ = xr.merge([flux, Rstokes])
        xStokes.append(xStokes_.assign_coords({'wavelength': float(wl) * 1000}))

    return xr.concat(xStokes, dim='wavelength')
