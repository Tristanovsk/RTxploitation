"""
Build the TOA look-up table (LUT) of OPAC aerosol models with wind speed
dimension (used by GRS v2).

The OSOAA ``osoaa_atmo_full`` simulations (atmosphere + rough sea surface,
black ocean) are gathered for:

- wind speeds ``winds`` (m/s),
- OPAC aerosol models ``models`` (with relative humidity ``rh``),
- aerosol optical thicknesses at 550 nm ``aots``,
- wavelengths ``wls`` (um),

into one :class:`xarray.Dataset` with dimensions
(wind, model, aot_ref, wl, sza, vza, azi) holding the aerosol optical
properties ``param_aerosol``, irradiance fluxes and the upward Stokes
vector (I, Q, U) at the level ``zidx`` (-1: TOA).

Input: OSOAA netCDF files in ``idir``
Output: netCDF LUT ``ofile``

Diagnostic plots of the LUT are available in ``lut_generator_aerosol_opac.py``
and ``notebook/lut_tool.ipynb``.
"""
import os
import numpy as np
import netCDF4 as nc
import xarray as xr

opj = os.path.join

# set absolute path
idir = '/data/vrtc/lut/atmo_full'

directions = ['up', 'down']
direction = directions[0]

# set -2 for BOA (or -1 for TOA)
zidx = [-1]
wls = [0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.6, 2.2, 2.5]
aots = ['0.0000', '0.001', '0.01', '0.1', '0.2', '0.35', '0.5', '0.7', '1.0', '1.5']
winds = ['0.5', '2', '4', '8', '12', '16']
rh = '_rh70'

models = ['ARCT','COAV', 'DESE', 'MACL', 'URBA']
file_type = 'osoaa_atmo_full'
# set aerosol parameters to be exported
param_aerosol = ['wl_ref','Cext_ref', 'ssa_ref',  'aot',
                 'Cext', 'ssa',  'asym_factor',    'bbp_ratio',
                 'bup_ratio', 'vol_mean_particle', 'nr']

# output LUT file
ofile = '/data/vrtc/xlut/toa_lut_opac_wind_up_v2.nc'

# ------------------------
# Load LUT
# ------------------------
xStokes = []
for wind in winds:
    print('wind ' + wind)
    xStokes_ = []
    for model in models:
        print(model)
        aer_model = model + rh
        xStokes__ = []
        for aot in aots:
            xwl = []
            for wl_ in wls:
                file = file_type + '_aot' + aot + '_aer' + aer_model + '_HR8.0_HA2.0_ws' + wind + '_wl{:0.3f}.nc'.format(
                    wl_)
                # print(file)

                pfile = opj(idir, file)
                # if not os.path.exists(pfile):
                #     print(pfile)
                #     continue
                with nc.Dataset(pfile) as ds:
                    sza = ds.variables['sza'][:]

                aerosol = xr.open_dataset(pfile, group='optical_properties/aerosol/'
                                          ).set_coords(['wl', 'aot_ref']).expand_dims(['wl', 'aot_ref'])[param_aerosol]

                Stokes=xr.open_dataset(pfile, group='stokes/' + direction
                                                  ).isel(z=zidx).expand_dims('aot_ref').assign_coords(
                        {'sza': sza, 'wl': aerosol.wl.values, 'aot_ref': aerosol.aot_ref.values, 'wind': float(wind)})

                flux = xr.open_dataset(pfile, group='flux').assign_coords(
                    {'sza': sza, 'wl': aerosol.wl.values, 'aot_ref': aerosol.aot_ref.values, 'wind': float(wind)}).sel(
                    z=Stokes.z)

                nan_num = np.isnan(Stokes.I.values).sum()
                if nan_num > 0:
                    print(nan_num, file)
                    # os.remove(pfile)
                merged = xr.merge([aerosol, flux, Stokes]).assign_coords({'model': aer_model}).squeeze().drop_vars('z')
                xwl.append(merged)

            xStokes__.append(xr.concat(xwl, dim='wl'))
        xStokes_.append(xr.concat(xStokes__, dim='aot_ref'))
    xStokes.append(xr.concat(xStokes_, dim='model'))
xlut = xr.concat(xStokes, dim='wind')

# ----------------------------------------
# save the LUT into netcdf
# ----------------------------------------
xlut.to_netcdf(ofile)
