"""
Build a netCDF LUT from OSOAA simulations for the OPAC aerosol models.

For each OPAC aerosol model (COAV, DESE, MACL, ...) and relative humidity,
the OSOAA outputs computed for a set of aerosol optical thicknesses (at 550 nm)
and wavelengths are gathered into one :class:`xarray.Dataset` with dimensions
(model, aot_ref, wl, z, sza, vza, azi). The dataset holds the aerosol optical
properties, the irradiance fluxes and the Stokes vector (I, Q, U) for the
chosen direction (``up`` by default) at the levels ``zidx``.

The second part of the script plots diagnostics of the LUT:
spectral aot, spectral Stokes parameters, normalized TOA radiance per model
and polar diagrams of I, Q, U and DoLP (per wavelength and per aerosol model).

Input: OSOAA netCDF files in ``idir``
Output: LUT netCDF file ``ofile`` and figures in ``odir``

See also ``lut_generator_grs_v2_opac.py`` and
``lut_generator_updown_wind_opac.py`` for LUTs including a wind-speed
dimension.
"""
import os
import numpy as np
import netCDF4 as nc
import xarray as xr

# ----------------------------
# set plotting styles
import cmocean as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

plt.ioff()

rc = {"font.family": "serif",
      "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 20})

# ----------------------------
from RTxploitation import lutplot

opj = os.path.join

# set absolute path
idir = '/sat_data/vrtc/lut/atmo'
odir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')

directions = ['down', 'up']
direction = directions[1]

zidx = [-2,-1]

aots = ['0.001','0.01', '0.1', '0.2','0.35', '0.5','0.7', '1.0','1.5']
# aots = ['0.5']

models = ['COAV', 'DESE', 'MACL', 'MAPO', 'COPO', 'URBA', 'ARCT', 'ANTA']
# file_type = 'osoaa_atmo'
# rhs = ['_rh0', '_rh70', '_rh90', '_rh99']
file_type = 'osoaa_water_bottom_albedo_1'
rhs = ['_rh70']
wls = [0.35, 0.400, 0.5, 0.6, 0.7, 0.8, 1, 1.6, 2.2, 2.5]
# output LUT file
ofile = '/DATA/git/satellite_app/hgrs/data/lut/opac_' + file_type + '_lut.nc'
# ------------------------
# Load LUT
# ------------------------
xStokes_ = []
for rh in rhs:
    for model in models:
        aer_model = model + rh
        xStokes__ = []
        for aot in aots:
            xlut = []
            for wl in wls:
                file = file_type+'_aot' + aot + '_aer{:s}_HR8.0_HA2.0_ws2_wl{:0.3f}.nc'.format(aer_model, wl)
                print(file)

                pfile = opj(idir, file)
                # if not os.path.exists(pfile):
                #     print(pfile)
                #     continue
                with nc.Dataset(pfile) as ds:
                    sza = ds.variables['sza'][:]

                aerosol = xr.open_dataset(pfile, group='optical_properties/aerosol/'
                                          ).set_coords(['wl', 'aot_ref']).expand_dims(['wl', 'aot_ref'])
                Stokes = xr.open_dataset(pfile, group='stokes/' + direction
                                         ).isel(z=zidx).expand_dims('aot_ref').assign_coords(
                    {'sza': sza, 'wl': aerosol.wl.values, 'aot_ref': aerosol.aot_ref.values})
                flux = xr.open_dataset(pfile, group='flux').assign_coords(
                    {'sza': sza, 'wl': aerosol.wl.values, 'aot_ref': aerosol.aot_ref.values}).sel(z=Stokes.z)
                nan_num = np.isnan(Stokes.I.values).sum()
                if nan_num > 0:
                    print(nan_num,file)
                    #os.remove(pfile)
                merged =  xr.merge([aerosol,flux, Stokes]).assign_coords({'model': aer_model})
                if merged.z[0] != 0:
                    # harmonize altitude coordinates (BOA, TOA) between files
                    print('unexpected z values, reset to [0, 300000] for', file)
                    merged = merged.assign_coords(z=[0, 300000])
                xlut.append(merged)

            xStokes__.append(xr.concat(xlut, dim='wl'))
        xStokes_.append(xr.concat(xStokes__, dim='aot_ref'))
xStokes = xr.concat(xStokes_, dim='model')


# ----------------------------------------
# save the LUT into netcdf
# ----------------------------------------
xStokes.to_netcdf(ofile)
# ----------------------------------------

xStokes['DoLP'] = (xStokes.Q ** 2 + xStokes.U ** 2) ** 0.5 / xStokes.I
xStokes['AoLP'] = np.degrees(np.sign(-xStokes.Q) * np.abs(np.arctan(-xStokes.U / xStokes.Q) / 2))
# ------------------------
# Plot LUT
# ------------------------
# --------------------------
# plot spectral aot
# --------------------------
fig, axs_ = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), sharey=True, sharex=True)
fig.subplots_adjust(bottom=0.1, top=0.98, left=0.075, right=0.98,
                    hspace=0.1, wspace=0.1)
axs = axs_.ravel()
# plt.figure(figsize=(10, 8))
colors = {'COAV': 'forestgreen', 'COPO': 'olivedrab', 'COCL': 'yellowgreen',
          'DESE': 'darkgoldenrod', 'MACL': 'cornflowerblue', 'MAPO': 'mediumblue', 'URBA': 'dimgrey',
          'ARCT': 'darkorchid', 'ANTA': 'palevioletred'}
irh = {'rh0': 0, 'rh70': 1, 'rh90': 2, 'rh99': 3}
x_ = xStokes.isel(aot_ref=1).squeeze()
for imodel, (model, x__) in enumerate(x_.groupby('model')):
    aer_type, rh = model.split('_')
    x__ = x__.squeeze()
    axs[irh[rh]].plot(x__.wl, x__.aot, 'o-', color=colors[aer_type], label=model)
    axs[irh[rh]].plot(x__.wl, x__.aot * x__.ssa, 'o--', color=colors[aer_type])  # , label='$aot_{sca}$')
for i in range(4):
    handles, labels = axs[i].get_legend_handles_labels()
    line1 = Line2D([0], [0], ls='-', label='extinction', color='k')
    line2 = Line2D([0], [0], ls='--', label='scattering', color='k')
    handles.extend([line1, line2])

    axs[i].legend(handles=handles, fontsize=12, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.99))
for i in range(2):
    axs_[i, 0].set_ylabel(r'$aot(\lambda )$')
    axs_[1, i].set_xlabel(r'$Wavelength\ (\mu m)$')
figfile = 'OPAC_spectralAOT'
plt.savefig(opj(odir, figfile + '.png'), dpi=300)

axs[0].semilogy()
for i in range(4):
    handles, labels = axs[i].get_legend_handles_labels()
    line1 = Line2D([0], [0], ls='-', label='extinction', color='k')
    line2 = Line2D([0], [0], ls='--', label='scattering', color='k')
    handles.extend([line1, line2])

    axs[i].legend(handles=handles, fontsize=11, ncol=2, loc='lower left', bbox_to_anchor=(0., 0.005))
plt.savefig(opj(odir, figfile + '_log.png'), dpi=300)
plt.close()

# --------------------------
# plot spectral Stokes terms
# --------------------------
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     "grey",  # 'forestgreen','yellowgreen',
                                                     "khaki", "gold",
                                                     'orangered', "firebrick", 'purple'])

norm = mpl.colors.Normalize(vmin=0.01, vmax=1)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
xStokes_ = xStokes.isel(z=-1, model=2)
model = str(xStokes_.model.values)
for sza in [10]:  # ,60
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    fig.subplots_adjust(bottom=0.1, top=0.98, left=0.05, right=0.98,
                        hspace=0.25, wspace=0.3)
    axs = axs.ravel()
    for ivza in [3]:  # ,20,34
        for azi in [90, ]:  # 45,90,135,180
            lut_ = xStokes_.isel(vza=ivza).sel(azi=azi, sza=sza).squeeze()
            vza = lut_.vza.values

            for i, param in enumerate(['I', 'Q', 'U', 'DoLP', 'AoLP']):
                for aot, lut__ in lut_.groupby('aot_ref'):
                    lut__[param].plot.line(x='wl', linestyle='--', marker='o', ax=axs[i], color=cmap(norm(aot)))

    for i in range(5):
        axs[i].minorticks_on()
        axs[i].set_xlabel(r'$Wavelength\ (\mu m)$')
        axs[i].set_title(None)
    axs[-1].axis('off')
    cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.04, location='top')
    cb.ax.tick_params(labelsize=22)
    cb.set_label(model + ' $aot(550nm)$', fontsize=22)

    figfile = 'Stokes_opac_' + model + '_sza' + str(sza) + '_vza' + str(vza) + '_azi' + str(azi)
    plt.suptitle(figfile)

    plt.savefig(opj(odir, figfile + '.png'), dpi=300)
    plt.close()

# --------------------------
# plot normalized radiance for different models

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     "grey",  # 'forestgreen','yellowgreen',
                                                     "khaki", "gold",
                                                     'orangered', "firebrick", 'purple'])

norm = mpl.colors.Normalize(vmin=0.01, vmax=1)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

ivza=3
sza=30
azi=90
xStokes_ = xStokes.isel(z=-1, model=2)
model = str(xStokes_.model.values)

# models_ = ['COAV','DESE','MACL','MAPO','COPO','URBA','ARCT','ANTA']

models_ = ['COAV', 'DESE', 'MACL', 'URBA', 'ARCT', 'ANTA']
rh = '_rh70'
param='I'
for sza in [10,30,60]:#,60
#     for ivza in [3]:  # ,20,34
    for azi in [0,90,180 ]:  # 45,90,135,180
        for rh in ['_rh70']:#'_rh0','_rh99']:
            xStokes_ = xStokes.isel(vza=ivza).sel(azi=azi, sza=sza).squeeze()
            vza = '{:.1f}'.format(xStokes_.vza.values)
            fig, axs_ = plt.subplots(nrows=2, ncols=3, figsize=(20, 10),sharex=True,sharey=True)
            fig.subplots_adjust(bottom=0.1, top=0.98, left=0.075, right=0.98,
                                hspace=0.05, wspace=0.05)
            axs = axs_.ravel()
            for imodel, model_ in enumerate(models_):
                model = model_ + rh
                lut_ = xStokes_.sel(model=model)


                for aot, lut__ in lut_.groupby('aot_ref'):
                    axs[imodel].plot(lut__.wl,lut__[param]/np.cos(np.radians(sza)), linestyle='--', marker='o',  color=cmap(norm(aot)))
                axs[imodel].set_title(model, y=1.0, pad=-18)
            for i in range(6):
                axs[i].minorticks_on()
            for i in range(3):
                axs_[1,i].set_xlabel(r'$Wavelength\ (\mu m)$')
            for i in range(2):
                axs_[i,0].set_ylabel(r'$L^{TOA}_n\ (sr^{-1})$')
            cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.04, location='top')
            cb.ax.tick_params(labelsize=22)
            cb.set_label(r' $aot(550nm)\ for\ $'+ rh, fontsize=22)

            figfile = 'Stokes_opac_radiance_sza' + str(sza) + '_vza' + vza + '_azi' + str(azi) + rh


            plt.savefig(opj(odir,'spectra', figfile + '.png'), dpi=300)
            plt.close()

# --------------------------
# Polar diagrams
# --------------------------
# by wavelength
lp = lutplot.plot()
nadir = False
# nadir=True
vzamax = 61  # 16 #61
suff = ''

if nadir:
    vzamax = 16
    suff = '_nadir'
szaidx = 3
szas = xStokes.sza.values
vza_all = xStokes.vza
wls_ = [0.4, 0.6, 0.8, 1.6]
Nrow = 4  # len(directions)
aot = '0.2'
aot_num = float(aot)
for model in xStokes.model.values:
    print(model)
    xStokes_ = xStokes.sel(model=model).sel(aot_ref=aot_num, method='nearest').isel(z=-1)
    figfile = os.path.join(odir, model + '_aot550_' + aot + '_SZA' + str(szas[szaidx]))
    fig, axs = plt.subplots(Nrow, 4, figsize=(20, 4 + Nrow * 4), subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(top=0.9)
    if Nrow == 1:
        axs = np.expand_dims(axs, axis=0)

    Stokes_ = xStokes_.isel(sza=szaidx).sel(wl=wls_, method='nearest').where(vza_all < vzamax, drop=True).squeeze()
    val = -0.05
    minmax = (np.min([*Stokes_.I]) * (1 - val), np.max([*Stokes_.I]) * (1 + val))
    minmaxQ = (np.min([*Stokes_.Q]) * (1 - val), np.max([*Stokes_.Q]) * (1 + val))
    minmaxU = (np.min([*Stokes_.U]) * (1 - val), np.max([*Stokes_.U]) * (1 + val))
    minmaxDOP = (0, 0.7)
    for iwl, wl in enumerate(wls_):  # print out group elements

        # ----------------------------
        # get dimensions

        vza = xStokes.vza.values
        if direction == 'down':
            vza = 180 - vza
        azi = xStokes.azi.values

        # ----------------------------
        # construct raster dimensions
        r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))

        # ----------------------------
        # slice and reshape arrays
        Stokes_ = xStokes_.sel(wl=wl, method='nearest').isel(sza=szaidx).where(vza_all < vzamax, drop=True).squeeze()
        I = Stokes_.I.T
        Q = Stokes_.Q.T
        U = Stokes_.U.T
        DOP = (Q ** 2 + U ** 2) ** 0.5 / I

        # ----------------------------
        # plot polar diagrams
        cmap = plt.cm.Spectral_r  # cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
        lp.add_polplot(axs[0, iwl], r, theta, I, title='wl={:.1f}'.format(wl) + r'$\mu m$', cmap=cmap, minmax=minmax)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
        lp.add_polplot(axs[1, iwl], r, theta, Q, title='Q(' + direction + ')', cmap=cmap, minmax=minmaxQ)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
        lp.add_polplot(axs[2, iwl], r, theta, U, title='U(' + direction + ')', cmap=cmap, minmax=minmaxU)
        cmap = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
        lp.add_polplot(axs[3, iwl], r, theta, DOP, title='DOP(' + direction + ')', cmap=cmap, colfmt='%0.2f',
                       minmax=minmaxDOP)

    plt.suptitle(model + ' aot550=' + aot + ' at SZA = ' + str(szas[szaidx]) + ' deg', **lp.font)
    plt.tight_layout()
    plt.savefig(figfile + suff + '.png', dpi=300, bbox_inches='tight')
    plt.close()

# --------------------------
# by aerosol models
# --------------------------
wls_ = [0.4, 0.6, 0.8, 1.6]
Nrow = 4  # len(directions)
szaidx = 1
aot = '0.2'
aot_num = float(aot)
for wl in wls_:

    xStokes_ = xStokes.sel(wl=wl, aot_ref=aot_num, method='nearest')
    figfile = os.path.join(odir, 'opac_aot550_' + aot + '_wl{:.0f}'.format(wl * 1000) + '_SZA' + str(szas[szaidx]))
    Ncol = len(xStokes.model)
    fig, axs = plt.subplots(Nrow, Ncol, figsize=(5 * Ncol, 4 + Nrow * 4), subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(top=0.9)
    if Nrow == 1:
        axs = np.expand_dims(axs, axis=0)
    Stokes_ = xStokes_.isel(sza=szaidx).where(vza_all < vzamax, drop=True).squeeze()
    val = -0.05
    minmax = (np.min([*Stokes_.I]) * (1 - val), np.max([*Stokes_.I]) * (1 + val))
    minmaxQ = (np.min([*Stokes_.Q]) * (1 - val), np.max([*Stokes_.Q]) * (1 + val))
    minmaxU = (np.min([*Stokes_.U]) * (1 - val), np.max([*Stokes_.U]) * (1 + val))
    minmaxDOP = (0, 0.7)
    for imodel, model in enumerate(xStokes.model.values):

        # ----------------------------
        # get dimensions

        vza = xStokes.vza.values
        if direction == 'down':
            vza = 180 - vza
        azi = xStokes.azi.values

        # ----------------------------
        # construct raster dimensions
        r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))

        # ----------------------------
        # slice and reshape arrays
        Stokes_ = xStokes_.sel(model=model).isel(sza=szaidx).where(vza_all < vzamax, drop=True).squeeze()
        I = Stokes_.I.T
        Q = Stokes_.Q.T
        U = Stokes_.U.T
        DOP = (Q ** 2 + U ** 2) ** 0.5 / I

        # ----------------------------
        # plot polar diagrams
        cmap = plt.cm.Spectral_r  # cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
        lp.add_polplot(axs[0, imodel], r, theta, I, title=model, cmap=cmap, minmax=minmax)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
        lp.add_polplot(axs[1, imodel], r, theta, Q, title='Q(' + direction + ')', cmap=cmap, minmax=minmaxQ)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
        lp.add_polplot(axs[2, imodel], r, theta, U, title='U(' + direction + ')', cmap=cmap, minmax=minmaxU)
        cmap = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
        lp.add_polplot(axs[3, imodel], r, theta, DOP, title='DOP(' + direction + ')', cmap=cmap, colfmt='%0.2f',
                       minmax=minmaxDOP)

    plt.suptitle('OPAC models, aot550=' + aot + ', wl={:.2f}'.format(wl) + r'$\mu m$ at SZA = ' + str(szas[szaidx]) + ' deg',
                 **lp.font)
    plt.tight_layout()
    plt.savefig(figfile + suff + '.png', dpi=300, bbox_inches='tight')
    plt.close()
