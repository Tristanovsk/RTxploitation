"""
Spectral signature of the GRS v1 bimodal (fine + coarse) aerosol models.

OSOAA atmosphere-only simulations are gathered for several fine-mode volume
fractions (CVfine), aerosol optical thicknesses (aot at 550 nm) and
wavelengths into an :class:`xarray.Dataset` (CVfine, aot_ref, wl, z, sza,
vza, azi). Two kinds of figures are produced:

- spectral extinction and scattering aot for each CVfine
  (``fig/grs_v1/GRS_v1_spectral_AOT.png``),
- normalized TOA radiance spectra for several geometries and CVfine
  (``fig/grs_v1/spectra/``).

For the polar diagrams and the check of the fine/coarse mixing
approximation, see ``plot_aerosol_mixture.py`` and
``check_ssa_approx_mixture.py``.
"""
import os
import numpy as np
import netCDF4 as nc
import xarray as xr

# ----------------------------
# set plotting styles
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

plt.ioff()
rc = {"font.family": "serif",
      "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 20})

opj = os.path.join


# ------------------------
# Load LUT
# ------------------------
# set absolute path
idir = '/sat_data/vrtc/lut/atmo'
odir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')
# other mixtures computed:
# pattern = "fine_rm0.10_sig0.40_nr1.48_ni-0.0035_coarse_rm0.81_sig0.60_nr1.48_ni-0.0035_HR8.0_HA2.0_ws2"
# pattern = "fine_rm0.10_sig0.40_nr1.48_ni-0.0000_coarse_rm0.81_sig0.60_nr1.48_ni-0.0000_HR8.0_HA2.0_ws2"
pattern = "fine_rm0.10_sig0.46_nr1.45_ni-0.0001_coarse_rm0.80_sig0.60_nr1.35_ni-0.0010_HR8.0_HA2.0_ws2"

directions = ['down', 'up']
direction = directions[1]
zidx = [-1]
CVfs = [0., 0.1, 0.2, .4,0.5, 0.6, 0.8,0.9,  1]
aots = ['0.01','0.1','0.2','0.5','1.0']
xStokes_=[]
for iCVf, CVf in enumerate(CVfs):
    xStokes__=[]
    for aot in aots:
        xlut = []
        for wl in [0.400, 0.5, 0.6, 0.7, 0.8, 1, 1.6, 2.2]:

            file = 'osoaa_atmo_aot' + aot + '_CVfine{:.2f}_'.format(CVf) + pattern + '_wl{:0.3f}.nc'.format(wl)

            print(file)

            pfile = opj(idir, file)

            with nc.Dataset(pfile) as ds:
                sza = ds.variables['sza'][:]
            aerosol = xr.open_dataset(pfile, group='optical_properties/aerosol/'
                                      ).set_coords(['wl','aot_ref']).expand_dims(['wl', 'aot_ref'])
            Stokes = xr.open_dataset(pfile, group='stokes/' + direction
                                     ).isel(z=zidx).expand_dims('aot_ref').assign_coords({'sza': sza,  'wl': aerosol.wl.values, 'aot_ref':aerosol.aot_ref.values}
                                                     )

            xlut.append(xr.merge([aerosol, Stokes]).assign_coords({'CVfine':CVf}))
        xStokes__.append(xr.concat(xlut, dim='wl'))
    xStokes_.append(xr.concat(xStokes__, dim='aot_ref'))
xStokes=xr.concat(xStokes_, dim='CVfine')
xStokes['DoLP']=(xStokes.Q**2+xStokes.U**2)**0.5/xStokes.I
xStokes['AoLP']=np.degrees(np.sign(-xStokes.Q)*np.abs(np.arctan(-xStokes.U/xStokes.Q)/2))

# ------------------------
# Plot LUT
# ------------------------

# --------------------------
# plot spectral aot
# --------------------------
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                        ['navy', "blue", 'lightskyblue',
                                                         'gray', 'yellowgreen', 'forestgreen','gold','darkgoldenrod']).reversed()

norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(16, 5),sharex=True)
fig.subplots_adjust(bottom=0.15, top=0.925, left=0.1, right=0.975,
                    hspace=0.1, wspace=0.25)
#axs = axs_.ravel()
# plt.figure(figsize=(10, 8))
x_ = xStokes.isel(aot_ref=1).squeeze()
for imodel,cvf in enumerate(CVfs):

    x__ = x_.sel(CVfine=cvf).squeeze()
    x__ = x__.squeeze()
    for i in range(2):
        axs[i].plot(x__.wl, x__.aot, 'o-', color=cmap(norm(cvf)), label='CVfine {:.1f}'.format(cvf))
        axs[i].plot(x__.wl, x__.aot * x__.ssa, 'o--', color=cmap(norm(cvf)))  # , label='$aot_{sca}$')
axs[1].semilogy()
#for i in range(4):
i=1
handles, labels = axs[i].get_legend_handles_labels()
line1 = Line2D([0], [0], ls='-', label='extinction', color='k')
line2 = Line2D([0], [0], ls='--', label='scattering', color='k')
handles.extend([line1, line2])
for i in range(2):
    axs[i].set_ylabel(r'$aot(\lambda )$')
    axs[i].set_xlabel(r'$Wavelength\ (\mu m)$')
axs[0].legend(handles=handles, fontsize=12, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.09))
axs[1].legend(handles=handles, fontsize=12, ncol=2)

figfile = 'GRS_v1_spectral_AOT'
plt.savefig(opj(odir,'grs_v1', figfile + '.png'), dpi=300)


# --------------------------
# plot normalized radiance for different models
CVfs_ = [0,0.1,0.4,0.6,0.9,1]
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     "grey",  # 'forestgreen','yellowgreen',
                                                     "khaki", "gold",
                                                     'orangered', "firebrick", 'purple'])

norm = mpl.colors.Normalize(vmin=0.01, vmax=1)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

ivza=3 # 8 # 3
sza=30
azi=90

param='I'
for sza in [10,30,60]:#,60
#     for ivza in [3]:  # ,20,34
    for azi in [0,90,180 ]:  # 45,90,135,180

        xStokes_ = xStokes.isel(vza=ivza).sel(azi=azi, sza=sza).squeeze()
        vza = '{:.1f}'.format(xStokes_.vza.values)
        fig, axs_ = plt.subplots(nrows=2, ncols=3, figsize=(20, 10),sharex=True,sharey=True)
        fig.subplots_adjust(bottom=0.1, top=0.98, left=0.075, right=0.98,
                            hspace=0.05, wspace=0.05)
        axs = axs_.ravel()
        for imodel,CVf in enumerate(CVfs_):

            lut_ = xStokes_.sel(CVfine=CVf)


            for aot, lut__ in lut_.groupby('aot_ref'):
                axs[imodel].plot(lut__.wl,lut__[param]/np.cos(np.radians(sza)), linestyle='--', marker='o',  color=cmap(norm(aot)))
            axs[imodel].set_title('CVfine={:.1f}'.format(CVf), y=1.0, pad=-18)
        for i in range(6):
            axs[i].minorticks_on()
        for i in range(3):
            axs_[1,i].set_xlabel(r'$Wavelength\ (\mu m)$')
        for i in range(2):
            axs_[i,0].set_ylabel(r'$L^{TOA}_n\ (sr^{-1})$')
        cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.04, location='top')
        cb.ax.tick_params(labelsize=22)
        cb.set_label(r' $aot(550nm)\ for$ sza=' + str(sza) + ', vza=' + vza + ', azi=' + str(azi), fontsize=22)
        figfile = 'Stokes_grs_v1_radiance_sza' + str(sza) + '_vza' + vza + '_azi' + str(azi)


        plt.savefig(opj(odir,'grs_v1','spectra', figfile + '.png'), dpi=300)
        plt.close()


