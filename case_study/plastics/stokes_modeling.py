import os

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib import gridspec

import matplotlib as mpl
from matplotlib import cm
import cmocean as cmo
import seaborn as sns

plt.rcParams.update({'font.family': 'Times New Roman',
                     'font.size': 22, 'axes.labelsize': 20,

                     })

rc = {"font.family": "serif",
      "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

from mpl_toolkits.axes_grid1 import make_axes_locatable

import cProfile
import lmfit as lm
import cmocean as cm

from RTxploitation import utils as u
import RTxploitation.load_lut_nc as load
from RTxploitation import lutplot

opj = os.path.join
lp = lutplot.plot()

odir = "/DATA/projet/garaba/OP3/nanomicro"
lutdir = "/media/harmel/vol1/Dropbox/work/git/vrtc/RTxploitation/study_cases/plastics/data"
plot = True

def arr_format(arr, fmt="{:0.3f}"):
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

    xStokes.to_netcdf(opj(lutdir,'lut_nr'+nr+'_rmed'+rmed+'.nc'))

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
plt.suptitle(figfile + ' at SZA = ' + str(sza) + ' deg', fontdict=lp.font)
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
model=0
psub = plt.subplot
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
                # gs = gridspec.GridSpec(2, 6)
                # gs.update(left=0.1, hspace=0.25,wspace=0.25)
                # axs = [psub(gs[0,0:2]),psub(gs[0,2:4]),psub(gs[0,4:6]),psub(gs[1,0:3]),psub(gs[1,3:6])]

                for i,param in  enumerate(['I', 'Q', 'U','DoLP','AoLP']):
                    for spm in lut_.spm_norm:
                        _ = lut_.sel(spm_norm=spm).reset_coords(['spm_norm'])
                        _[param].plot.line(x='wavelength',color=cmap(norm(spm )), linestyle='--',marker='o',ax=axs[i])

                for i in range(5):
                    axs[i].minorticks_on()
                    axs[i].set_xlabel('$Wavelength\ (nm)$')
                    axs[i].set_title(None)
                axs[-1].axis('off')
                cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30,pad=0.08, location='top')
                cb.ax.tick_params(labelsize=22)
                cb.set_label('Microplastic $(g\ m^{-3})$', fontsize=22)
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
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15),sharex=True,sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.94, left=0.086, right=0.9,
                    hspace=0.1, wspace=0.05)
for isza,sza in enumerate([1,3,6]):
    xStokes.I.isel(z=z,vza=3,sza=sza,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,0])
    xStokes.Q.isel(z=z,vza=3,sza=sza,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,1])
    xStokes.U.isel(z=z,vza=3,sza=sza,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,2])
plt.show()


Trans = xStokes.isel(z=-1)/xStokes.isel(z=0)

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15),sharex=True,sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.94, left=0.086, right=0.9,
                    hspace=0.1, wspace=0.05)
for isza,sza in enumerate([1,3,6]):
    Trans.I.isel(vza=3,sza=sza,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,0])
    Trans.Q.isel(vza=3,sza=sza,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,1])
    Trans.U.isel(vza=3,sza=sza,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,2])
plt.show()

df = u.data().load()
construct_netcdf = False  # True #
if construct_netcdf:

    models = ('fine', 'medium', 'coarse')
    Rrs, Qxr, Uxr, sed_props = load.load_osoaa(IQU=True)

    params = ['Cext', 'Cscat', 'ssa', 'asym_factor',
              'bbp_ratio', 'bup_ratio', 'vol_mean_particle', 'nr', 'ni', 'reff',
              'vareff', 'max_depth', 'a_w', 'b_w', 'chl',
              'a_phy', 'b_phy', 'a_cdom440', 'S_cdom440', 'a_cdom',
              'a_cdim440', 'S_cdim440', 'a_cdim', 'Cext_phy', 'Cscat_phy',
              'ssa_phy', 'asym_factor_phy', 'bbp_ratio_phy', 'bup_ratio_phy',
              'vol_mean_particle_phy', 'nr_phy', 'ni_phy', 'reff_phy', 'vareff_phy']

    iop = sed_props[params].drop_duplicates().droplevel([1, 3])
    iop['Cscat_vol'] = iop.Cscat / iop.vol_mean_particle
    iop = iop.to_xarray()
    lut_data = xr.merge([Rrs.rename('Rrs'), Qxr.rename('Q'), Uxr.rename('U'), iop])
    lut_data.to_netcdf('data/lut_sediment_' + lut + '.nc')
else:
    lut_data = xr.open_dataset('data/lut_sediment_' + lut + '.nc')

Rrs = lut_data.Rrs
iop = lut_data[list(lut_data.keys())[3:]]

DOP = (lut_data.Q ** 2 + lut_data.U ** 2) ** 0.5 / Rrs
lut_data['DOP'] = DOP

if lut == 'pure':
    models, wls, seds, zs, szas, vzas, azis = Rrs.coords.values()
else:
    wls, seds, models, zs, szas, vzas, azis = Rrs.coords.values()

density = 1  # 2.65 #1  # 2.6  # 2.65
sed = Rrs.sed.values  # np.logspace(-2,np.log10(600),80)
sed_norm = sed / density


# -------------------------
# add scattering angle values
def scat_angle(sza, vza, azi):
    '''
    self.azi: azimuth in rad for convention azi=0 when sun-sensor in oppositioon
    :return: scattering angle in deg
    '''
    print(sza, vza, azi)
    sza = np.pi / 180. * sza
    vza = np.pi / 180. * vza
    azi = np.pi / 180. * azi
    ang = -np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(azi)
    # ang = np.cos(np.pi - sza) * np.cos(vza) - np.sin(np.pi - sza) * np.sin(vza) * np.cos(azi)
    ang = np.arccos(ang) * 180 / np.pi

    return ang


scat_geom = scat_angle(Rrs.sza, Rrs.vza, Rrs.azi)
lut_data['scat_ang'] = scat_geom

g1, g2 = 0.089, 0.125
u = np.linspace(0, 0.8, 101)


# -------------------------
# Compute for all relative sediment concentration:
# - ssa, single scattering albedo
# - back_ssa, bb /(a+bb)
# - asym, asymmetry parameter
# -------------------------

def ssa(prop, sed_norm, df_pandas=False):
    b_sed = prop.Cscat_vol * sed_norm
    a_sed = prop.Cext / prop.vol_mean_particle * sed_norm - b_sed

    a_tot = prop.a_w + prop.a_phy + prop.a_cdom + prop.a_cdim + a_sed
    bb_tot = prop.b_phy * prop.bbp_ratio_phy + prop.b_w * 0.5 + b_sed * prop.bbp_ratio
    b_tot = prop.b_phy + prop.b_w + b_sed

    asym = (prop.b_phy * prop.asym_factor_phy + b_sed * prop.asym_factor) / b_tot
    ssa = b_tot / (a_tot + b_tot)
    back_ssa = bb_tot / (a_tot + bb_tot)
    b_a = b_tot / a_tot
    bb_a = bb_tot / a_tot
    asymb_a = asym * b_tot / a_tot
    if df_pandas:
        return pd.DataFrame({'ssa': ssa.T.to_pandas().stack(),
                             'back_ssa': back_ssa.T.to_pandas().stack(),
                             'asym': asym.T.to_pandas().stack()})
    else:
        return xr.Dataset({'ssa': ssa, 'back_ssa': back_ssa,
                           'bb_a': bb_a, 'b_a': b_a, 'asymb_a': asymb_a,
                           'asym': asym})


iops = []
for ised, sed_ in enumerate(seds):
    iops.append(ssa(iop, sed_).expand_dims('sed'))
iops = xr.concat(iops, dim='sed')

aop_iop = xr.merge([lut_data[['Rrs', 'Q', 'U', 'DOP']], iops])




param = 'DOP'
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(28, 8))
for ilevel in [0, 1]:
    aop_iop.isel(z=ilevel, model=1, sza=1, vza=2, azi=3).plot.scatter(
        'ssa', param, hue='wl', cmap='Spectral_r', ax=axs[ilevel, 0])
    aop_iop.isel(z=ilevel, model=1, sza=1, vza=2, azi=3).plot.scatter(
        'back_ssa', param, hue='wl', cmap='Spectral_r', ax=axs[ilevel, 1])
    aop_iop.isel(z=ilevel, model=1, sza=1, vza=2, azi=3).plot.scatter(
        'b_a', param, hue='wl', cmap='Spectral_r', ax=axs[ilevel, 2])
    aop_iop.isel(z=ilevel, model=1, sza=1, vza=2, azi=3).plot.scatter(
        'bb_a', param, hue='wl', cmap='Spectral_r', ax=axs[ilevel, 3])
    aop_iop.isel(z=ilevel, model=1, sza=1, vza=2, azi=3).plot.scatter(
        'asymb_a', param, hue='wl', cmap='Spectral_r', ax=axs[ilevel, 4])
plt.tight_layout()
plt.show()
plt.savefig(opj(odir, 'forward_model', param + '_behavior.png'), dpi=300)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
vzas[2]
for ilevel in [0, 1]:
    aop_iop.isel(z=ilevel, model=1, vza=2, azi=3).plot.scatter(
        'ssa', param, hue='sza', cmap='Spectral_r', ax=axs[ilevel, 0], alpha=0.6)

    aop_iop.isel(z=ilevel, model=1, vza=2, azi=3).plot.scatter(
        'b_a', param, hue='sza', cmap='Spectral_r', ax=axs[ilevel, 1], alpha=0.6)

plt.tight_layout()
plt.savefig(opj(odir, 'forward_model', param + '_vs_bssa_bb_a.png'), dpi=300)

ilevel = 0
for param in ('Rrs', 'Q', 'U', 'DOP'):
    for var in ('b_a', 'ssa', 'bb_a', 'back_ssa', 'asymb_a', 'asym'):

        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

        azi = 90  # 180
        for i, vza in enumerate([0, 30, 60]):
            aop_iop.isel(z=ilevel, model=1).interp(vza=vza, azi=azi).plot.scatter(
                var, param, hue='sza', cmap='Spectral_r', ax=axs[i, 0], alpha=0.6)

            aop_iop.isel(z=ilevel, model=2).interp(vza=vza, azi=azi).plot.scatter(
                var, param, hue='sza', cmap='Spectral_r', ax=axs[i, 1], alpha=0.6, )
            aop_iop.isel(z=ilevel, model=0).interp(vza=vza, azi=azi).plot.scatter(
                var, param, hue='sza', cmap='Spectral_r', ax=axs[i, 2], alpha=0.6, )
        axs[0, 0].set_title(aop_iop.model.values[1])
        axs[0, 1].set_title(aop_iop.model.values[2])
        axs[0, 2].set_title(aop_iop.model.values[0])

        plt.tight_layout()
        plt.savefig(opj(odir, 'forward_model', param + '_vs_' + var + '_level' + str(ilevel) + '.png'), dpi=300)
        plt.close()

szas = aop_iop.sza

model = 'm7'
aop_iop.isel(z=ilevel).sel(model=model, sza=szas[szas < 80]).interp(vza=[0, 10, 30, 60], azi=[0, 90, 180]).plot.scatter(
    'b_a', param, hue='sza', row='vza', col="azi", cmap='Spectral_r', alpha=0.6, aspect=1.5)
plt.savefig(opj(odir, 'forward_model', param + '_vs_bb_a_geom_' + model + '.png'), dpi=300)

x, y = 'ssa', 'asymb_a'
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
axs = axs.ravel()
iops.plot.scatter(x, y, hue='wl', cmap='Spectral_r', ax=axs[0])
iops.plot.scatter(x, y, hue='sed', cmap='Spectral_r', ax=axs[1])
iops.plot.scatter(x, y, hue='model', cmap='Spectral_r', ax=axs[2])
iops.plot.scatter(x, y, hue='asym', cmap='Spectral_r', ax=axs[3])
plt.tight_layout()
plt.savefig(opj(odir, 'forward_model', y + '_vs_' + x + '_' + lut + '.png'), dpi=300)


# --------------------------
# fit Rrs = f(u)

def linear(B, x):
    '''Linear function y = m*x + b'''
    return B[0] + B[1] * x


def exponential(B, x):
    # return B[0] - (B[0]-B[1]) * np.exp(-B[2] * x)
    return B[0] * np.exp(B[1] * x) + B[2]


def hyperbolic(B, x, b=1):
    return B[1] / (1 + B[0] * x) ** (1 / b)


def poly4(B, x):
    return B[0] * x + B[1] * x ** 2 + B[2] * x ** 3 + B[3] * x ** 4


def poly2(B, x):
    return B[0] * x + B[1] * x ** 2  # + B[3]*x**3


def fit(Rrs, omega):
    model = poly2

    # model = poly4

    def objfunc(x, Rrs, omega):
        '''
        Objective function to be minimized
        :param x: vector of unknowns
        :param theta: scattering angle
        :param vsf: phase function
        '''
        gs = np.array(list(x.valuesdict().values()))
        simu = model(gs, omega)
        return Rrs - simu

    pars = lm.Parameters()

    pars.add('g1', value=0.089)  # , min=0, max=1)
    pars.add('g2', value=0.125)  # , min=0, max=1)
    # pars.add('g3', value=0.125)
    # pars.add('g4', value=0.125)
    return lm.Minimizer(objfunc, pars, fcn_args=(Rrs, omega)), model


cmap = plt.cm.Spectral_r
norm = mpl.colors.Normalize(vmin=50, vmax=170)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

for color, lims in (('wl', [500, 1000]), ('ssa', [0.5, 1]), ('asym', [0.7, 1])):
    pass
print(color)
color = "scatt_ang"
fig, axs = plt.subplots(ncols=3, figsize=(20, 8), sharex=True, sharey=True)
fig.subplots_adjust(bottom=0.1, top=0.85, left=0.07, right=0.95,
                    hspace=0.1, wspace=0.05)
cmap = plt.cm.Spectral_r
level = 1
res = []
params = ('sub_rrs', 'Rrs')
for im, model in enumerate(models):
    iops_ = iops.sel(model=model)
    for isza, sza in enumerate(szas):
        print(sza.values)
        for azi in azis:  # [range(0,37,9)]: #azis[range(0,37,3)]:
            print(azi)
            for vza in vzas[range(0, 61, 5)]:  # [[0,10,20,30,40]]:#.sel(wl=[490.,  560.,  665.,  780.,  865., 1020.])
                Rrs_ = Rrs.isel(z=level).sel(model=model, sza=sza, azi=azi,
                                             vza=vza)  # ssa(iop.sel(wl=wls_),sed_,df_pandas=True)
                x_sorted = iops_.stack(dim=("wl", 'sed'))
                omega_b = x_sorted.back_ssa
                Rrs_data = Rrs_.stack(dim=("wl", 'sed')).sortby(omega_b)
                omega_b = omega_b.sortby(omega_b)
                x_sorted = x_sorted.sortby(omega_b)
                min1, func = fit(Rrs_data, omega_b)
                # cProfile.run("out1 = min1.least_squares()", sort=1)
                out1 = min1.least_squares(method='lm', xtol=1e-5, ftol=1e-5)
                x = out1.x
                # out1.params.pretty_print()
                res.append([str(model.values), float(sza), float(azi), float(vza), x[0], x[1], out1.redchi])

                p = axs[im].plot(omega_b, Rrs_data, 'o', color=cmap(norm(scat_geom.sel(sza=sza, vza=vza, azi=azi))),
                                 ms=5, alpha=0.6)
                #                p=axs[im].scatter(omega_b,Rrs_data,c=x_sorted[color],vmin=lims[0], vmax=lims[1], cmap=cmap)
                axs[im].plot(u, func(x, u), '-', color=cmap(norm(scat_geom.sel(sza=sza, vza=vza, azi=azi))), alpha=0.6,
                             lw=1)

    axs[im].set_xlabel(r'$b_b / (a + b_b) $')
    axs[im].set_title(model.values)
    if level == 0:
        axs[im].set_ylim([-0.01, 0.2])
        axs[im].plot(u, (g1 * u + g2 * u ** 2), '--', c='black', lw=2.5, label='QAA', alpha=0.86)
        axs[im].plot(u, (0.0949 * u + 0.0794 * u ** 2), '-', c='black', lw=2.5, label='Gordon88', alpha=0.86)
        axs[im].plot(u, (0.089 * u + 0.17 * u ** 2), '-.', c='black', lw=2.5, label='Lee99', alpha=0.86)
    else:
        axs[im].set_ylim([-0.01, 0.08])
        axs[im].plot(u, (0.045 * u + 0.08 * u ** 2), '-.', c='black', lw=2.5, label='first approx', alpha=0.6)
if level == 0:
    axs[0].set_ylabel(r'$r_{rs}\ (sr^{-1})$')
    axs[0].legend()
else:
    axs[0].set_ylabel(r'$R_{rs}\ (sr^{-1})$')

# plt.colorbar(p,label=color)
cb = fig.colorbar(sm, ax=axs, shrink=0.6, aspect=30, panchor=(0.5, 0.25), location='top')
cb.set_label('Scattering Angle (deg)', fontsize=25)

plt.savefig(opj(odir, 'forward_model', params[level] + '_vs_bssa_fit_' + color + '_' + lut + '.png'), dpi=300)

res_df = pd.DataFrame(res, columns=['model', 'sza', 'azi', 'vza', 'g1', 'g2', 'redchi'])
res_xr = res_df.set_index(['model', 'sza', 'vza', 'azi']).to_xarray()

sns.pairplot(res_df, hue='sza', palette='Spectral_r')
for sza in [10, 30, 60]:
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(top=0.9,
                        hspace=0.1, wspace=0.2)
    res_xr_int = res_xr.interp(vza=np.linspace(0, 60, 100))  # ,azi=np.linspace(0,360,100))
    r, theta = np.meshgrid(res_xr_int.vza, np.radians(res_xr_int.azi))
    for i, model in enumerate(res_xr_int.model):
        rast = res_xr_int.sel(model=model, sza=sza)
        lp.add_polplot(axs[i, 0], r, theta, rast.g1.T, title='g1', cmap=cmap)
        lp.add_polplot(axs[i, 1], r, theta, rast.g2.T, title='g2', cmap=cmap)
        lp.add_polplot(axs[i, 2], r, theta, rast.redchi.T, title='redchi', cmap=cmap)
    plt.savefig(opj(odir, 'forward_model', 'polar_gordon_param_sza' + str(sza) + '.png'), dpi=300)

plt.show()

# -------------------------
levels = ('(0-)', '(0+)', '(TOA)')

cmap = plt.cm.Spectral_r
norm = mpl.colors.Normalize(vmin=sed_norm[0], vmax=sed_norm[-1])
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 15), sharey=True)
color = 'sed'
for im, model in enumerate(models):
    for iwl, wl in enumerate(wls):
        iops_ = iops.sel(wl=wl, model=model)
        for isza, sza in enumerate(szas):
            for azi in [0, 90, 180]:
                for vza in [0, 15, 40]:
                    Rrs_ = Rrs.sel(model=model, wl=wl, sza=sza).interp(azi=azi,
                                                                       vza=vza)  # ssa(iop.sel(wl=wls_),sed_,df_pandas=True)

                    axs[0, im].scatter(iops_.back_ssa, Rrs_.isel(z=0), c=iops[color], vmin=0, vmax=400, cmap=cmap,
                                       alpha=0.6)
                    axs[1, im].scatter(iops_.back_ssa, Rrs_.isel(z=1), c=iops[color], vmin=0, vmax=400, cmap=cmap,
                                       alpha=0.6)
                    axs[2, im].scatter(iops_.ssa, Rrs_.isel(z=1), c=iops[color], vmin=0, vmax=400, cmap=cmap, alpha=0.6)

    axs[0, im].plot(u, (g1 * u + g2 * u ** 2), '--', c='grey', alpha=0.6)
plt.show()

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
fig.subplots_adjust(bottom=0.15, top=0.92, left=0.1, right=0.9,
                    hspace=0.3, wspace=0.2)
color = 'sza'
norm = mpl.colors.Normalize(vmin=0, vmax=60)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
for im, model in enumerate(models):
    axs[0, im].set_title(model.values)
    for iwl, wl in enumerate(wls):
        iops_ = iops.sel(wl=wl, model=model)
        for isza, sza in enumerate([0, 20, 40, 60]):
            Rrs_ = Rrs.sel(model=model, wl=wl, sza=sza)
            DOP_ = DOP.sel(model=model, wl=wl, sza=sza)
            for azi in [90]:
                for vza in [0, 15, 40]:
                    _ = Rrs_.interp(azi=azi, vza=vza)
                    __ = DOP_.interp(azi=azi, vza=vza)

                    p = axs[0, im].plot(iops_.back_ssa, __.isel(z=0), '--o', color=cmap(norm(sza)), mec='grey', ms=7,
                                        alpha=0.6)
                    axs[1, im].plot(iops_.back_ssa, _.isel(z=0), '--o', color=cmap(norm(sza)), mec='grey', ms=7,
                                    alpha=0.6)
                    axs[2, im].plot(iops_.back_ssa, _.isel(z=1), '--o', color=cmap(norm(sza)), mec='grey', ms=7,
                                    alpha=0.6)

    axs[1, im].plot(u, (g1 * u + g2 * u ** 2), '--', c='grey', alpha=0.6)
    axs[1, im].plot(u, (0.0949 * u + 0.0794 * u ** 2), ':', c='grey', alpha=0.6)
    axs[2, im].plot(u, (0.03 * u + 0.1 * u ** 2), ':', c='grey', alpha=0.6)

    axs[-1, im].set_xlabel('$\omega_b$')
axs[0, 0].set_ylabel('DOP')
axs[1, 0].set_ylabel('rrs')
axs[2, 0].set_ylabel('Rrs')
# fig.tight_layout(rect=(0, 0, 1, 0.98))

cb = fig.colorbar(sm, ax=axs, shrink=0.6, location='top')
cb.set_label('Sun Zenith Angle (deg)')
plt.savefig(opj(odir, 'forward_model', 'Rrs_vs_bssa_' + lut + '.png'), dpi=300)
plt.show()

# plot u
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 5), sharex=True, sharey=True)
for ised, sed_ in enumerate(seds):
    iop_ = ssa(iop.sel(wl=wls), sed_)
    for im, model in enumerate(models):
        axs[im].plot(iop_.wl, iop_.back_ssa.sel(model=model), '--o', color=cmap(norm(sed_)), mec='grey', ms=7,
                     alpha=0.6)
        if ised == 0:
            axs[im].set_title(model.data)
            axs[im].set_xlabel('Wavelength (nm)')
axs[0].set_ylabel('$\omega_b$')
fig.tight_layout(rect=(0, 0, 1, 0.98))
plt.savefig(opj(odir, 'forward_model', 'bssa_vs_wl_' + lut + '.png'), dpi=300)

markers = ['-d', '--o', ':*']
inc = [5, 0, -5]
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), sharex=True, sharey=True)
for ised, sed_ in enumerate(seds):
    iop_ = ssa(iop.sel(wl=wls), sed_)
    for im, model in enumerate(models):
        axs.plot(iop_.wl + inc[im], iop_.back_ssa.sel(model=model), markers[im], color=cmap(norm(sed_)), mec='grey',
                 ms=6, alpha=0.6)
        if ised == 0:
            axs.set_xlabel('Wavelength (nm)')
axs.set_ylabel('$\omega_b$')
fig.tight_layout(rect=(0, 0, 1, 0.98))
plt.savefig(opj(odir, 'forward_model', 'bssa_vs_wl_allinone_' + lut + '.png'), dpi=300)
