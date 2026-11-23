import os, sys
import glob
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
# sys.path.extend(['/home/harmel/Dropbox/work/VRTC/OSOAA_profile/exe/lut_package'])
from RTxploitation import lutplot

opj = os.path.join


# ------------------------
# Load LUT
# ------------------------
# set absolute path
idir = '/sat_data/vrtc/lut/atmo'
odir = '/DATA/git/vrtc/RTxploitation/study_cases/aerosol/fig'
pattern = "fine_rm0.10_sig0.40_nr1.48_ni-0.0035_coarse_rm0.81_sig0.60_nr1.48_ni-0.0035_HR8.0_HA2.0_ws2"
pattern = "fine_rm0.10_sig0.40_nr1.48_ni-0.0000_coarse_rm0.81_sig0.60_nr1.48_ni-0.0000_HR8.0_HA2.0_ws2"
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

            lut = nc.Dataset(pfile)

            wl = lut.getncattr('OSOAA.Wa')
            sza = lut.variables['sza'][:]
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
    axs[i].set_ylabel('$aot(\lambda )$')
    axs[i].set_xlabel('$Wavelength\ (\mu m)$')
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
            axs_[1,i].set_xlabel('$Wavelength\ (\mu m)$')
        for i in range(2):
            axs_[i,0].set_ylabel('$L^{TOA}_n\ (sr^{-1})$')
        cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, pad=0.04, location='top')
        cb.ax.tick_params(labelsize=22)
        cb.set_label(' $aot(550nm)\ for$ sza=' + str(sza) + ', vza=' + vza + ', azi=' + str(azi), fontsize=22)
        #plt.show()

        figfile = 'Stokes_grs_v1_radiance_sza' + str(sza) + '_vza' + vza + '_azi' + str(azi)


        plt.savefig(opj(odir,'grs_v1','spectra', figfile + '.png'), dpi=300)
        plt.close()


lp = lutplot.plot()
nadir=False
vzamax = 61 #16 #61
suff=''
if nadir:
    vzamax = 16
    suff='_nadir'
szaidx = 3

Nrow = 4  # len(directions)
if False:
    figfile = os.path.join(odir, pattern)
    fig, axs = plt.subplots(Nrow, 6, figsize=(30, 4 + Nrow * 4), subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(top=0.9)
    if Nrow == 1:
        axs = np.expand_dims(axs, axis=0)
    CVfs_ = [ 0.1, 0.2, .4, 0.6, 0.8, 0.9]
    Stokes_ = lut.isel(sza=szaidx).sel(CVfine=CVfs_).where(Stokes.vza < vzamax, drop=True)
    val=0.
    minmax = (np.min([*Stokes_.I]) * (1 - val), np.max([*Stokes_.I]) * (1 + val))
    minmaxQ = (np.min([*Stokes_.Q]) * (1 - val), np.max([*Stokes_.Q]) * (1 + val))
    minmaxU = (np.min([*Stokes_.U]) * (1 - val), np.max([*Stokes_.U]) * (1 + val))
    minmaxDOP=(0,0.7)
    for iCVf, CVf in enumerate(CVfs_):  # print out group elements

        # ----------------------------
        # get dimensions

        vza = Stokes.vza
        if direction == 'down':
            vza = 180 - vza
        azi = Stokes.azi

        # ----------------------------
        # construct raster dimensions
        r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))

        # ----------------------------
        # slice and reshape arrays
        Stokes_ = lut.sel(CVfine=CVf).isel(sza=szaidx).where(Stokes.vza < vzamax, drop=True)
        I = Stokes_.I.T
        Q = Stokes_.Q.T
        U = Stokes_.U.T
        DOP = (Q ** 2 + U ** 2) ** 0.5 / I

        # ----------------------------
        # plot polar diagrams
        cmap = plt.cm.Spectral_r  # cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
        lp.add_polplot(axs[0, iCVf], r, theta, I, title='CVf={:.2f}'.format(CVf), cmap=cmap, minmax=minmax)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
        lp.add_polplot(axs[1, iCVf], r, theta, Q, title='Q(' + direction + ')', cmap=cmap, minmax=minmaxQ)
        cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
        lp.add_polplot(axs[2, iCVf], r, theta, U, title='U(' + direction + ')', cmap=cmap, minmax=minmaxU)
        cmap = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
        lp.add_polplot(axs[3, iCVf], r, theta, DOP, title='DOP(' + direction + ')', cmap=cmap, colfmt='%0.2f', minmax=minmaxDOP)

    plt.suptitle(os.path.basename(file) + ' at SZA = ' + str(sza[szaidx]) + ' deg', fontdict=lp.font)
    plt.tight_layout()
    plt.savefig(figfile + '_sza' + str(sza[szaidx]) + suff +'.png', dpi=300, bbox_inches='tight')
    plt.close()

#--------------------------
# plot aerosol scattering matrix terms
#--------------------------


cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen','gold','darkgoldenrod']).reversed()

norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(16, 5), sharex=True)
fig.subplots_adjust(bottom=0.15, top=0.925, left=0.1, right=0.975,
                    hspace=0.1, wspace=0.25)
axs = axs.ravel()
for cvf in lut.CVfine.values:
    lut_ = lut.sel(CVfine=cvf).squeeze()
    axs[0].plot(lut_.scatt_ang,lut_.F11,color=cmap(norm(cvf)),label='CVfine {:.1f}'.format(cvf))
    axs[1].plot(lut_.scatt_ang,lut_.F12,color=cmap(norm(cvf)))
    axs[2].plot(lut_.scatt_ang,lut_.F33,color=cmap(norm(cvf)))

axs[0].set_title('$P_{11}$')
axs[1].set_title('$P_{12}$')
axs[2].set_title('$P_{33}$')
for i in range(3):
    axs[i].set_xlabel('$Scattering\ angle\ (deg)$')
axs[0].semilogy()

axs[0].legend( loc='upper right',fontsize=11)
figfile = os.path.join(odir, 'scat_mat','aerosol_scattering_matrix')
plt.tight_layout()
plt.savefig(figfile +  '_wl' + wl + '.png', dpi=300)


#--------------------------
# plot fine/coarse mixture approximation
#--------------------------
Cext = lut.Cext_ref.sel(CVfine=[1,0]).squeeze().values #np.array([0.017, 10.5])

# Cext=np.array([0.05,9.812])
ssa = np.array([0.98, 0.877])
Csca = ssa * Cext

aot = 0.2
# --------------------------
# Size distrib param
rn_med_f, sigma_f = 0.1, 0.4
rn_med_c, sigma_c = 0.81, 0.6


def V0_lognorm(rn_med=0.1, sigma=0.4):
    return 4 / 3 * np.pi * np.exp(3 * np.log(rn_med) + 4.5 * sigma**2)


V0_f = V0_lognorm(rn_med=rn_med_f, sigma=sigma_f)
V0_c = V0_lognorm(rn_med=rn_med_c, sigma=sigma_c)

aot_ref = 0.2

plt.figure()
CVf= np.linspace(0,1,100)
Nnorm = (CVf / V0_f + (1 - CVf) / V0_c)
eta_f = CVf / V0_f / Nnorm
Cext_mix = eta_f * Cext[0] + (1 - eta_f) * Cext[1]
gamma = eta_f * Cext[0] / Cext_mix


lut.Cext_ref.plot()
plt.semilogy()
plt.plot(CVf,Cext_mix)
plt.figure()
plt.plot(CVf,gamma)
plt.plot(CVf,eta_f)
plt.ylabel('gamma')
plt.show()

for CVf in [0., 0.1, 0.5, 0.9, 1]:
    Nnorm = (CVf / V0_f + (1 - CVf) / V0_c)
    eta_f = CVf / V0_f / Nnorm
    eta = np.array([eta_f, 1 - eta_f])
    Cext_mix = eta_f * Cext[0] + (1 - eta_f) * Cext[1]  # (CVf/V0_f *Cext[0] + (1-CVf)/V0_c * Cext[1]) / Nnorm
    N0 = aot_ref / Cext_mix
    gamma = eta * Cext / Cext_mix  # * CVf/V0_f/ (CVf/V0_f+(1-CVf)/V0_c)
    # eta = eta/np.sum(eta)
    print(CVf, Nnorm, Cext_mix, gamma)

# * Cext[0] / aot_

nlayers=45
cmap_diff = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
CVf = 0.9
szaidx = 1
for szaidx in [1, 3, 6]:
    for CVf in [0.1, 0.2, 0.5, 0.8, 0.9]:
        Nnorm = (CVf / V0_f + (1 - CVf) / V0_c)
        eta_f = CVf / V0_f / Nnorm
        eta = np.array([eta_f, 1 - eta_f])
        Cext_mix = eta_f * Cext[0] + (1 - eta_f) * Cext[1]
        gamma = eta * Cext / Cext_mix

        Stokes_c = lut.sel(CVfine=0).isel(sza=szaidx).where(lut.vza < vzamax, drop=True)
        Stokes_f = lut.sel(CVfine=1).isel(sza=szaidx).where(lut.vza < vzamax, drop=True)
        Stokes_ = lut.sel(CVfine=CVf).isel(sza=szaidx).where(lut.vza < vzamax, drop=True)

        # ----------------------------
        # construct raster dimensions
        vza=Stokes_.vza.values
        azi=Stokes_.azi.values
        r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))
        I = Stokes_.I.T
        If = Stokes_f.I.T
        Ic = Stokes_c.I.T
        val = 0.
        minmax = (np.min([*Ic, *If]) * (1 - val), np.max([*Ic, *If]) * (1 + val))

        fig, axs = plt.subplots(2, 3, figsize=(14, 10), subplot_kw=dict(projection='polar'))
        fig.subplots_adjust(hspace=0.25, wspace=0.05)
        cmap = plt.cm.Spectral_r  # cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
        lp.add_polplot(axs[0, 0], r, theta, Ic, title='CVf={:.2f}'.format(0), nlayers=nlayers,cmap=cmap, minmax=minmax, scale=False)
        lp.add_polplot(axs[0, 1], r, theta, I, title='CVf={:.2f}'.format(CVf),  nlayers=nlayers,cmap=cmap, minmax=minmax, scale=False)
        lp.add_polplot(axs[0, 2], r, theta, If, title='CVf={:.2f}'.format(1), nlayers=nlayers,cmap=cmap, minmax=minmax, scale=False)
        Imix = gamma[0] * If + gamma[1] * Ic
        cax = lp.add_polplot(axs[1, 1], r, theta, Imix, title='Reconstructed', nlayers=nlayers,cmap=cmap,
                             minmax=minmax, scale=False)
        lp.add_polplot(axs[1, 2], r, theta, (Imix-I)/I, title='rel. diff.', nlayers=nlayers,cmap=cmap_diff )
        axs[1, 0].set_axis_off()
        axs[1, -1].set_axis_off()
        plt.colorbar(cax, ax=axs[1, 0], format='%0.1e', pad=-0.3, fraction=0.05)

        figfile = os.path.join(odir, 'compar_aerosol_mode_mixture')
        plt.tight_layout()
        plt.savefig(figfile + '_sza' + str(sza[szaidx]) + '_CVfine{:.2f}'.format(CVf) + '_wl' + wl + '.png', dpi=300)
        plt.close()

plt.show()
