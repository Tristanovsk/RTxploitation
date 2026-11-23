"""
Polarized radiance (Stokes vector) of waters containing microplastics.

OSOAA simulations computed for several normalized concentrations of
microplastic particles (``spm_norm``, g m-3) with a given refractive index
(``nr``) and median radius (``rmed``, um) are gathered into a netCDF LUT
(``data/lut_nr<nr>_rmed<rmed>.nc``, built from the OSOAA outputs with
:func:`RTxploitation.load_lut_nc.load_osoaa` if not present).

Figures:

- polar diagrams of I, Q, U, DoLP and AoLP at BOA for several concentrations,
- spectra of I, Q, U, DoLP and AoLP for several geometries and levels,
- spectra of the Stokes parameters and of the TOA/BOA ratio for several
  solar zenith angles.

Output: figures in ``<odir>/fig``

The fitting of Rrs-IOP relationships formerly appended to this script is now
in ``invRrs.lut_factory`` (``aop_iop_fitting.py``).
"""
import os

import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean as cm

plt.rcParams.update({'font.size': 22, 'axes.labelsize': 20})

rc = {"font.family": "serif",
      "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

import RTxploitation.load_lut_nc as load
from RTxploitation import lutplot

opj = os.path.join
lp = lutplot.plot()

odir = "/DATA/projet/garaba/OP3/nanomicro"
lutdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def arr_format(arr, fmt="{:0.3f}"):
    '''Format numbers into a list of strings (as used in OSOAA file names).'''
    return [fmt.format(x) for x in arr]

# ---------------------------------
# Load and arrange data
# ---------------------------------
spm_norms_ = [0.01, 0.1,0.3, 0.5, 1., 2.00,5., 10., 50.] #150., 200., 300., 400., 600.]
spm_norms = arr_format(spm_norms_, "{:0.2f}")

nr='1.15'
rmed='1000.0'
wl_= np.arange(400,851,25)/1e3
wls = arr_format(wl_, "{:0.3f}")

lutfile=opj(lutdir,'lut_nr'+nr+'_rmed'+rmed+'.nc')
if os.path.exists(lutfile):
    xStokes = xr.open_dataset(lutfile)
else:
    xStokes=[]
    for spm_norm in spm_norms:
        labels = []
        for wl in wls:
            labels.append('__sed%s_nr%s_rmed%s_wl%s'%(spm_norm,nr,rmed,wl))
        x_ = load.load_osoaa(labels=labels,water_signal=True)
        xStokes.append(x_.assign_coords({'spm_norm': float(spm_norm)}))
    xStokes = xr.concat(xStokes, dim='spm_norm')

    xStokes.to_netcdf(lutfile)

xStokes['DoLP']=(xStokes.Q**2+xStokes.U**2)**0.5/xStokes.I
xStokes['AoLP']=np.degrees(np.sign(-xStokes.Q)*np.abs(np.arctan(-xStokes.U/xStokes.Q)/2))

# -------------------------
# PLOTTING SECTION
# -------------------------
levels = ['BOA', 'TOA']
ilevel = 0
wl = 500
sza = 30
xStokes_ = xStokes.sel(wavelength=wl, sza=sza).isel(z=ilevel)
conc = 'water'
vzamax = 61
azi_, vza_ = np.linspace(0, 360, 721), np.linspace(0, vzamax, 101)
r, theta = np.meshgrid(vza_, np.radians(azi_))
xStokes_ = xStokes_.interp(azi=azi_, method='cubic').interp(vza=vza_, method='cubic')
xStokes_['DoLP'] = (xStokes_.Q ** 2 + xStokes_.U ** 2) ** 0.5 / xStokes_.I
xStokes_['AoLP'] = np.degrees(np.sign(-xStokes_.Q) * np.abs(np.arctan(-xStokes_.U / xStokes_.Q) / 2))

fig, axs = plt.subplots(3, 5, figsize=(29, 13), subplot_kw=dict(projection='polar'))
fig.subplots_adjust(top=0.9)

for i, ispm_norm in enumerate([0, 3, 5]):
    x_ = xStokes_.isel(spm_norm=ispm_norm)
    print(x_.spm_norm.values)
    conc = str(x_.spm_norm.values) + '$g/m^3$'
    # ----------------------------
    # plot polar diagrams
    cmap = cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
    lp.add_polplot(axs[i, 0], r, theta, x_.I.T, title='I(' + conc + ')', cmap=cmap)
    cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
    lp.add_polplot(axs[i, 1], r, theta, x_.Q.T, title='Q(' + conc + ')', cmap=cmap)
    cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
    lp.add_polplot(axs[i, 2], r, theta, x_.U.T, title='U(' + conc + ')', cmap=cmap)
    cmap_dop = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
    lp.add_polplot(axs[i, 3], r, theta, x_.DoLP.T, title='DoLP(' + conc + ')', cmap=cmap_dop, colfmt='%0.2f')

    # cmap = mpl.colors.LinearSegmentedColormap.from_list("",
    #                                             ['navy', "blue", 'lightskyblue',
    #                                              "grey", 'forestgreen', 'yellowgreen',
    #                                              "white", "gold", "darkgoldenrod",
    #                                              'orangered', "firebrick", 'purple'], N=100)
    cmax = int(xStokes_.AoLP.__abs__().max() * 100) / 100
    lp.add_polplot(axs[i, 4], r, theta, x_.AoLP.T, title='AoLP(' + conc + ')', cmap=cmap, colfmt='%0.2f', vmin=-cmax,
                   vmax=cmax)
figfile = 'Stokes_lut_nr' + nr + '_rmed' + rmed + '_level' + levels[ilevel] + '_wl' + str(wl)
plt.suptitle(figfile + ' at SZA = ' + str(sza) + ' deg', **lp.font)
plt.tight_layout()
plt.savefig(opj(odir, 'fig', figfile + '_sza' + str(sza) + '.png'), dpi=300, bbox_inches='tight')
plt.close()



# colors=['black','orangered',  'purple']
# azi=45
# for ilevel in [0,1]:
#     lut_ = xStokes.isel(z=ilevel, sza=3, vza=6,spm_norm=5).sel(azi=azi)
#     fig, axs = plt.subplots( figsize=(5, 5))
#     for i,param in  enumerate(['I', 'Q', 'U']):
#             lut_[param].plot.line(x='wavelength',linestyle='--',marker='o',color=colors[i],ax=axs)
#     axs.set_xlabel('$Wavelength\ (nm)$')
#     #plt.tight_layout(rect=[0, 0.03, 1, 0.8])
#     plt.savefig(opj(odir, 'fig', 'Stokes_lut_nr'+nr+'_rmed'+rmed+'_azi'+str(azi)+'_level'+str(ilevel) +'.png'), dpi=300)

# --------------------------
# plot spectra
# --------------------------
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                    ['navy',"blue",'lightskyblue',
                     "grey", #'forestgreen','yellowgreen',
                        "khaki", "gold",
                     'orangered', "firebrick", 'purple'])

norm = mpl.colors.LogNorm(vmin=0.01, vmax=50)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
for ilevel in [0,1]:
    for sza in [30,60]:
        for ivza in [3,20,34]:
            for azi in [0,45,90,135,180]:
                lut_ = xStokes.isel(z=ilevel, vza=ivza).sel(azi=azi,sza=sza)
                vza = lut_.vza.values
                fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
                fig.subplots_adjust(bottom=0.1, top=0.98, left=0.05, right=0.98,
                                    hspace=0.25, wspace=0.3)
                axs=axs.ravel()
                for i,param in  enumerate(['I', 'Q', 'U','DoLP','AoLP']):
                    for spm in lut_.spm_norm:
                        _ = lut_.sel(spm_norm=spm).reset_coords(['spm_norm'])
                        _[param].plot.line(x='wavelength',color=cmap(norm(spm )), linestyle='--',marker='o',ax=axs[i])

                for i in range(5):
                    axs[i].minorticks_on()
                    axs[i].set_xlabel(r'$Wavelength\ (nm)$')
                    axs[i].set_title(None)
                axs[-1].axis('off')
                cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30,pad=0.08, location='top')
                cb.ax.tick_params(labelsize=22)
                cb.set_label(r'Microplastic $(g\ m^{-3})$', fontsize=22)
                figfile = 'Stokes_lut_nr'+nr+'_rmed'+rmed+'_sza'+str(sza)+'_vza'+str(vza)+'_azi'+str(azi)+'_level'+str(ilevel)
                plt.suptitle(figfile)

                plt.savefig(opj(odir, 'fig', figfile+'.png'), dpi=300)
                plt.close()
plt.show()
azi=90
lut_ = xStokes.isel(z=0, sza=3, vza=6).sel(wavelength=500.,azi=azi) #.interp(spm_norm=np.linspace(0.01,50.,100),method='cubic')
fig, ax = plt.subplots(figsize=(5, 5))
lut_.I.plot(linestyle='--',marker='o',color='black',ax=ax)
plt.semilogx()

z=0
ispm = 5  # index of the microplastic concentration (spm_norm) shown in the spectra below
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15),sharex=True,sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.94, left=0.086, right=0.9,
                    hspace=0.1, wspace=0.05)
for isza,sza in enumerate([1,3,6]):
    xStokes.I.isel(z=z,vza=3,sza=sza,spm_norm=ispm,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,0])
    xStokes.Q.isel(z=z,vza=3,sza=sza,spm_norm=ispm,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,1])
    xStokes.U.isel(z=z,vza=3,sza=sza,spm_norm=ispm,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,2])
plt.show()


Trans = xStokes.isel(z=-1)/xStokes.isel(z=0)

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15),sharex=True,sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.94, left=0.086, right=0.9,
                    hspace=0.1, wspace=0.05)
for isza,sza in enumerate([1,3,6]):
    Trans.I.isel(vza=3,sza=sza,spm_norm=ispm,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,0])
    Trans.Q.isel(vza=3,sza=sza,spm_norm=ispm,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,1])
    Trans.U.isel(vza=3,sza=sza,spm_norm=ispm,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,2])
plt.show()

