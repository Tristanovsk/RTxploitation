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

plt.ioff()
plt.rcParams.update({'font.size': 16})

rc = {"font.family": "serif",
      "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

plt.rcParams.update({'font.size': 18, 'axes.labelsize': 22})

# ----------------------------
# sys.path.extend(['/home/harmel/Dropbox/work/VRTC/OSOAA_profile/exe/lut_package'])
from RTxploitation import lutplot

opj = os.path.join

# set absolute path
idir = '/sat_data/vrtc/lut/atmo'
odir = '/DATA/git/vrtc/RTxploitation/study_cases/aerosol/fig'

xlut = []
for wl in [0.400,0.5,0.6,0.7,0.8,1,1.6,2.2]:

    pattern = "fine_rm0.10_sig0.40_nr1.48_ni-0.0035_coarse_rm0.81_sig0.60_nr1.48_ni-0.0035_HR8.0_HA2.0_ws2_wl{:0.3f}".format(wl)

    directions = ['down', 'up']
    direction = directions[1]

    zidx = -1
    if zidx == -1:
        direction = 'up'

    # ------------------------
    # Load LUT
    # ------------------------

    CVfs = [0., 0.1, 0.2, .4, 0.5, 0.6, 0.8, 0.9, 1]
    for iCVf, CVf in enumerate(CVfs):
        file = "osoaa_atmo_aot0.2_CVfine{:.2f}_".format(CVf) + pattern + '.nc'
        pfile = opj(idir, file)

        lut = nc.Dataset(pfile)

        wl = float(lut.getncattr('OSOAA.Wa'))
        sza = lut.variables['sza'][:]
        aerosol = xr.open_dataset(pfile, group='optical_properties/aerosol/'
                                  ).set_coords('wl').expand_dims(['wl', 'CVfine']).assign_coords({'CVfine': [CVf]})

        Stokes = xr.open_dataset(pfile, group='stokes/' + direction
                                       ).isel(z=zidx).expand_dims(['wl','CVfine']
                                                ).assign_coords({'sza': sza, 'CVfine': [CVf],'wl':aerosol.wl})


        xlut.append(xr.merge([aerosol,Stokes]))
lut = xr.merge(xlut)

# ------------------------
# Plot LUT
# ------------------------
lp = lutplot.plot()
nadir=False
vzamax = 61 #16 #61
suff=''
if nadir:
    vzamax = 16
    suff='_nadir'
szaidx = 3

Nrow = 4  # len(directions)

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen','gold','darkgoldenrod']).reversed()

norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])


#--------------------------
# plot norm. radiance spectra
#--------------------------
isza,ivza,iazi = 3,5,30
lut_view = lut.isel(sza=isza,vza=ivza,azi=iazi)
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(13, 5), sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.925, left=0.1, right=0.975,
                    hspace=0.1, wspace=0.25)
axs = axs.ravel()
for cvf in lut_view.CVfine.values:
    lut_ = lut_view.sel(CVfine=cvf).squeeze()
    axs[0].plot(lut_.wl,lut_.I,color=cmap(norm(cvf)),label='CVfine {:.1f}'.format(cvf))
    axs[1].plot(lut_.wl,lut_.I,color=cmap(norm(cvf)),label='CVfine {:.1f}'.format(cvf))
axs[0].set_xlim([0.4,0.9])
for i in range(2):
    axs[i].set_xlabel('$Wavelength\ (\mu m)$')
    axs[i].set_ylabel('$Normalized\ radiance\ (-)$')

axs[0].legend( loc='upper right',fontsize=11)
figfile = os.path.join(odir, 'spectra','radiance')
plt.tight_layout()
plt.savefig(figfile +'_sza'+  str(lut_view.sza.values) + '.png', dpi=300)


#--------------------------
# plot extinction aot
#--------------------------

fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(16, 5),sharex=True)
fig.subplots_adjust(bottom=0.15, top=0.925, left=0.1, right=0.975,
                    hspace=0.1, wspace=0.25)
scale = 0.5

for cvf in lut.CVfine.values:
    lut_ = lut.sel(CVfine=cvf).squeeze()
    axs[0].plot(lut_.wl,scale*lut_.aot,'o-',color=cmap(norm(cvf)),label='CVfine {:.1f}'.format(cvf))
    axs[1].plot(lut_.wl,scale*lut_.aot,'o-',color=cmap(norm(cvf)),label='CVfine {:.1f}'.format(cvf))
axs[1].semilogy()
for i in range(2):
    axs[i].set_xlabel('$Wavelength\ (\mu m)$')
    axs[i].set_ylabel('$aot(\lambda )$')
    axs[i].minorticks_on()
axs[1].legend( loc='lower left',fontsize=12)
figfile = os.path.join(odir, 'spectra','aerosol_optical_thickness_grs_v1')
plt.tight_layout()
plt.savefig(figfile  + '.png', dpi=300)

#--------------------------
# plot aerosol scattering matrix terms
#--------------------------

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
