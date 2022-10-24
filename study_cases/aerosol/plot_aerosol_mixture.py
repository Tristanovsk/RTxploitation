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

# set absolute path
idir = '/sat_data/vrtc/lut/atmo'
odir = '/DATA/git/vrtc/RTxploitation/study_cases/aerosol/fig'
pattern = "fine_rm0.10_sig0.40_nr1.48_ni-0.0035_coarse_rm0.81_sig0.60_nr1.48_ni-0.0035_HR8.0_HA2.0_ws2_wl0.865"

directions = ['down', 'up']
direction = directions[1]

zidx = -1
if zidx == -1:
    direction = 'up'

# ------------------------
# Load LUT
# ------------------------

wl = []
xStokes = []
CVfs = [0., 0.1,0.2, .4,0.5, 0.6, 0.8,0.9, 1]
for iCVf, CVf in enumerate(CVfs):
    file = "osoaa_atmo_aot0.2_CVfine{:.2f}_".format(CVf) + pattern + '.nc'
    pfile = opj(idir, file)

    lut = nc.Dataset(pfile)

    wl = lut.getncattr('OSOAA.Wa')
    sza = lut.variables['sza'][:]
    xStokes.append(xr.open_dataset(pfile, group='stokes/' + direction
                                   ).assign_coords({'sza': sza, 'CVfine': CVf}).isel(z=zidx))
Stokes = xr.concat(xStokes, dim='CVfine')

# ------------------------
# Plot LUT
# ------------------------
lp = lutplot.plot()
vzamax = 61
szaidx = 3

Nrow = 4  # len(directions)

figfile = os.path.join(odir, pattern)
fig, axs = plt.subplots(Nrow, 6, figsize=(30, 4 + Nrow * 4), subplot_kw=dict(projection='polar'))
fig.subplots_adjust(top=0.9)
if Nrow == 1:
    axs = np.expand_dims(axs, axis=0)
for iCVf, CVf in enumerate(CVfs):  # print out group elements

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
    Stokes_ = Stokes.sel(CVfine=CVf).isel(sza=szaidx).where(Stokes.vza < vzamax, drop=True)
    I = Stokes_.I.T
    Q = Stokes_.Q.T
    U = Stokes_.U.T
    DOP = (Q ** 2 + U ** 2) ** 0.5 / I

    # ----------------------------
    # plot polar diagrams
    cmap = plt.cm.Spectral_r  # cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
    lp.add_polplot(axs[0, iCVf], r, theta, I, title='CVf={:.2f}'.format(CVf), cmap=cmap, minmax=(0.035, 0.09))
    cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
    lp.add_polplot(axs[1, iCVf], r, theta, Q, title='Q(' + direction + ')', cmap=cmap)
    cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
    lp.add_polplot(axs[2, iCVf], r, theta, U, title='U(' + direction + ')', cmap=cmap)
    cmap = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
    lp.add_polplot(axs[3, iCVf], r, theta, DOP, title='DOP(' + direction + ')', cmap=cmap, colfmt='%0.2f')

plt.suptitle(os.path.basename(file) + ' at SZA = ' + str(sza[szaidx]) + ' deg', fontdict=lp.font)
plt.tight_layout()
plt.savefig(figfile + '_sza' + str(sza[szaidx]) + '.png', dpi=300, bbox_inches='tight')
plt.close()



Cext=np.array([0.017,10.5])

#Cext=np.array([0.05,9.812])
ssa=np.array([0.98,0.877])
Csca = ssa * Cext

aot=0.2
# --------------------------
# Size distrib param
rn_med_f, sigma_f = 0.1, 0.4
rn_med_c, sigma_c = 0.81, 0.6
def V0_lognorm(rn_med=0.1, sigma=0.4):
    return 4 / 3 * np.pi * np.exp(3 * np.log(rn_med) + 4.5 * sigma * 2)

V0_f = V0_lognorm(rn_med=rn_med_f, sigma=sigma_f)
V0_c = V0_lognorm(rn_med=rn_med_c, sigma=sigma_c)
aot_ = CVf/V0_f * ssa[0]*Cext[0] + (1-CVf)/V0_c * ssa[1]*Cext[1]
aot_ref = 0.2

for CVf in [0.,0.1,0.5,0.9,1]:
    Nnorm = (CVf/V0_f+(1-CVf)/V0_c)
    eta_f = CVf/V0_f / Nnorm
    eta = np.array([eta_f,1-eta_f])
    Cext_mix = eta_f * Cext[0] + (1-eta_f) * Cext[1] #(CVf/V0_f *Cext[0] + (1-CVf)/V0_c * Cext[1]) / Nnorm
    N0 = aot_ref / Cext_mix
    gamma = eta *Cext /Cext_mix #* CVf/V0_f/ (CVf/V0_f+(1-CVf)/V0_c)
    #eta = eta/np.sum(eta)
    print(CVf,Nnorm,Cext_mix,gamma)

 #* Cext[0] / aot_

CVf=0.9
szaidx = 1
for szaidx in [1,3,6]:
    for CVf in [0.1,0.2,0.5,0.8,0.9]:

        Nnorm = (CVf/V0_f+(1-CVf)/V0_c)
        eta_f = CVf/V0_f / Nnorm
        eta = np.array([eta_f,1-eta_f])
        Cext_mix = eta_f * Cext[0] + (1-eta_f) * Cext[1]
        gamma = eta*Cext/Cext_mix

        Stokes_c = Stokes.sel(CVfine=0).isel(sza=szaidx).where(Stokes.vza < vzamax, drop=True)
        Stokes_f = Stokes.sel(CVfine=1).isel(sza=szaidx).where(Stokes.vza < vzamax, drop=True)
        Stokes_ = Stokes.sel(CVfine=CVf).isel(sza=szaidx).where(Stokes.vza < vzamax, drop=True)



        I = Stokes_.I.T
        If = Stokes_f.I.T
        Ic = Stokes_c.I.T
        val=0.
        minmax = (np.min([*Ic,*If]) * (1-val),np.max([*Ic,*If]) * (1+val))

        fig, axs = plt.subplots(2, 3, figsize=(14, 10), subplot_kw=dict(projection='polar'))
        fig.subplots_adjust(hspace=0.25, wspace=0.05)
        cmap = plt.cm.Spectral_r  # cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
        lp.add_polplot(axs[0, 0], r, theta, Ic, title='CVf={:.2f}'.format(0), cmap=cmap, minmax=minmax,scale=False)
        lp.add_polplot(axs[0, 1], r, theta, I, title='CVf={:.2f}'.format(CVf), cmap=cmap, minmax=minmax,scale=False)
        lp.add_polplot(axs[0, 2], r, theta, If, title='CVf={:.2f}'.format(1), cmap=cmap, minmax=minmax,scale=False)

        cax = lp.add_polplot(axs[1, 1], r, theta, gamma[0]*If+gamma[1]*Ic, title='Reconstructed', cmap=cmap, minmax=minmax,scale=False)

        axs[1,0].set_axis_off()
        axs[1,-1].set_axis_off()
        plt.colorbar(cax, ax=axs[1,-1], format='%0.1e', pad=0.1, fraction=0.034)
        figfile = os.path.join(odir, 'compar_aerosol_mode_mixture')
        plt.tight_layout()
        plt.savefig(figfile + '_sza' + str(sza[szaidx]) + '_CVfine{:.2f}'.format(CVf)+ '_wl'+wl+'.png', dpi=300)
        plt.close()


plt.show()
