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

# TODO check sunglint component from OSOAA check OSOAA_SOS_CORE_harmel.F line 2181 or 3982


from RTxploitation import lutplot
from RTxploitation import utils as u

idir = '/home/harmel/VRTC/lut/sediment_lnd/'  # /DATA/projet/VRTC/lut/bagré'
odir = '/DATA/projet/bagre/sediment/fig/'

wl_num = [0.665, 0.865]
wl_list = u.misc.arr_format(wl_num, "{:0.3f}")
levels=['below','above','TOA']
sed_ = [2.00, 10., 20., 50., 100., 150., 200., 300., 400., 600.]
# sed_=[2.00, 20., 100., 400.]

sed_list = u.misc.arr_format(sed_, "{:0.2f}")
models = ['fine', 'medium', 'coarse']
mode1_rm = u.misc.arr_format([0.06, 1.0, 14.], "{:0.2f}")
mode1_sig = u.misc.arr_format([0.4, 0.7, 0.5], "{:0.2f}")
mode1_nr = u.misc.arr_format([1.17, 1.17, 1.17], "{:0.2f}")
mode1_ni = u.misc.arr_format([-1e-3, -1e-3, -1e-3], "{:0.4f}")
mode1_rate = u.misc.arr_format([1, 1, 1], "{:0.2f}")

mode2_rm = u.misc.arr_format([0.06, 1.0, 14.], "{:0.2f}")
mode2_sig = u.misc.arr_format([0.4, 0.7, 0.5], "{:0.2f}")
mode2_nr = u.misc.arr_format([1.17, 1.17, 1.17], "{:0.2f}")
mode2_ni = u.misc.arr_format([-1e-3, -1e-3, -1e-3], "{:0.4f}")
mode2_rate = u.misc.arr_format([0, 0, 0], "{:0.2f}")

pattern = 'osoaa_tot_aot0.1_aero_rg0.10_sig0.46_nr1.45_ni-0.0010_ws2_chl3.00_'
model_name = '_sedmod1_rm%s_sig%s_mr%s_mi%s_rate%s_sedmod2_rm%s_sig%s_mr%s_mi%s_rate%s'

directions = ['up', 'down']
direction = directions[0]
lp = lutplot.plot()
vzamax = 61
szaidx = 3
zidx = 1

Nrow = len(directions)
wl_ = []
z_slice = -3  # to keep the last three z-layers, i.e., subsurface, above-surface (BOA), top-of-atmosphere (TOA)

pattern = 'osoaa_tot_aot0.1_aero_rg0.80_sig0.60_nr1.45_ni-0.0010_ws2_chl3.00_sed*_sedjs*_sednr1.17_sedni0.0_wl0.665.nc'
files = sorted(glob.glob(os.path.join(idir, pattern)))


def s2f(list):
    return np.array(list).astype(np.float)


def _toxr(arr):
    arr = np.array(arr)
    return xr.DataArray(arr,
                        dims=('wl', 'sed', 'model', 'z', 'sza', 'vza', 'azi'),
                        coords={'wl': wl_num, 'sed': s2f(sed_list), 'model': models, 'z': z, 'sza': sza,
                                'vza': vza, 'azi': azi})


wl_ = []
for file in files:
    # f=file.split('_')
    lut = nc.Dataset(file)

    # if you want to check netcdf content (e.g., groups):
    u.nc_utils().print_groups(lut)

    wl_.append(float(lut.getncattr('OSOAA.Wa')) * 1e3)
    sza = lut.variables['sza'][:]

sed_props = pd.DataFrame()
first = True
figfile = 'Rrs_sednr1.17_LND'
for i, model in enumerate(models):
    pattern = model_name % (mode1_rm[i], mode1_sig[i], mode1_nr[i], mode1_ni[i], mode1_rate[i],
                            mode2_rm[i], mode2_sig[i], mode2_nr[i], mode2_ni[i], mode2_rate[i])

    for iwl, wl in enumerate(wl_list):
        for ised, sed in enumerate(sed_list):
            # _wl0.665.nc

            file_ = 'osoaa_tot_aot0.1_aero_rg0.10_sig0.46_nr1.45_ni-0.0010_ws2_chl3.00_sed' + sed + pattern + '_wl%s.nc' % wl
            # print(file_)
            file = os.path.join(idir, file_)
            if not os.path.exists(file):
                print('Warning %s does not exist' % file)
                continue
            lut = nc.Dataset(file)

            sed_prop = lut['/optical_properties/sediment'].variables
            keys = list(sed_prop.keys())
            ds = pd.DataFrame(sed_prop['F11'][:], columns=['vsf'], index=sed_prop['scatt_ang'][:])
            ds['wl'] = float(wl)
            ds['sed'] = float(sed)
            ds['model'] = model
            for key in keys:
                ds[key] = sed_prop[key][:]

            sed_props = pd.concat([sed_props, ds])

            # if you want to check netcdf content (e.g., groups):
            u.nc_utils().print_groups(lut)

            wl_.append(float(lut.getncattr('OSOAA.Wa')) * 1e3)
            sza = lut.variables['sza'][:]

            # ----------------------------
            # get data group
            stokes = lut['stokes/' + direction]

            # ----------------------------
            # initialize arrays within first loop
            if first:
                Nlut = stokes.variables['I'][z_slice:].shape
                dim = (len(wl_list), len(sed_list), len(models)) + Nlut
                Rrs, RQ, RU, I, Q, U = u.misc.init_array(6, dim)
                first = False

            # ----------------------------
            # get dimensions
            z = stokes.variables['z'][z_slice:]
            vza = stokes.variables['vza'][:]
            azi = stokes.variables['azi'][:]

            # ----------------------------
            # get data values
            if direction == 'down':
                vza = 180 - vza
            I[iwl, ised, i, ...] = stokes.variables['I'][z_slice:] / np.pi
            Q[iwl, ised, i, ...] = stokes.variables['Q'][z_slice:] / np.pi
            U[iwl, ised, i, ...] = stokes.variables['U'][z_slice:] / np.pi

            for isza, sza_ in enumerate(sza):
                print(sza_)
                mu0 = np.cos(np.pi / 180 * sza_)
                Rrs[:, :, :, :, isza, ...] = I[:, :, :, :, isza, ...] / mu0
                RQ[:, :, :, :, isza, ...] = Q[:, :, :, :, isza, ...] / mu0
                RU[:, :, :, :, isza, ...] = U[:, :, :, :, isza, ...] / mu0

Ixr, Qxr, Uxr = _toxr(Rrs), _toxr(RQ), _toxr(RU)

# ===============================
#       PLOTTING SECTION
# ===============================
# ---------------------------------
# plot vsf/optical data
# ---------------------------------
colors = ["pale red", "amber", "greyish", "faded green", "dusty purple"] * 4

sed_props.sort_values(by='model', inplace=True)
sed_props.sort_index(inplace=True)
sed_props['legend'] = '$n_i = $' + sed_props.ni.apply(lambda x: '{0:.0e}'.format(-x)) + '$; g = $' + \
                      sed_props.asym_factor.apply(lambda x: '{0:.3f}'.format(x)) + '$; b_{bp}/b_p = $' + \
                      sed_props.bbp_ratio.apply(lambda x: '{0:.4f}'.format(x)) + '$; ssa = $' + \
                      sed_props.ssa.apply(lambda x: '{0:.2f}'.format(x)) + '$; r_{eff} = $' + \
                      sed_props.reff.apply(lambda x: '{0:.3f}'.format(x))
sed_props_sub = sed_props[sed_props.sed == float(sed_list[0])]

# ---------------------------------
# plot VSF
# ---------------------------------
sns.set(style="ticks", color_codes=True)
g = sns.FacetGrid(sed_props_sub, col="wl", row="model", hue="legend", palette='tab20c')  # sns.xkcd_palette(colors))

g.map(plt.semilogy, 'scatt_ang', 'vsf', linewidth=2)
for ax in g.axes.ravel():
    h, l = ax.get_legend_handles_labels()
    neworder = sorted(range(len(l)), key=l.__getitem__)
    l = [l[i] for i in neworder]
    h = [h[i] for i in neworder]

    ax.legend(h, l)
g.fig.set_size_inches(14, 16)
g.savefig(opj(odir, 'vsf', 'VSF_used.pdf'))

# ---------------------------------
# plot Rrs data monodirectional
# ---------------------------------


# one direction plots
zidx = 0
vzaidx = 27
aziidx = 18
import proplot as plot

for szaidx in (1, 3, 5):
    sza_ = str(sza[szaidx])
    plot.rc.cycle = 'qual2'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    g = Ixr.isel(sza=szaidx, azi=aziidx, vza=vzaidx, z=[0, 1, 2]).plot.line(x='sed', marker='o', row='z', hue='model',
                                                                            col='wl')
    g.set_xlabels('SPM in mg/L')
    g.set_ylabels('$R_{rs}\ (sr^{-1})$')
    plt.suptitle('SZA = ' + sza_ + ' deg, VZA = ' + str(vza[vzaidx]) + ' deg, AZI = ' + str(azi[aziidx]),
                 fontdict=lp.font)
    plt.tight_layout(rect=[0.0, 0., 0.9, 0.9])

    plt.savefig(opj(odir, 'onedir', figfile + '_sza' + sza_ + '_onedir.png'), dpi=300)
    plt.close('all')



    # ----------------------------

# ---------------------------------
# plot Rrs diagrams
# ---------------------------------

    # construct raster dimensions
    r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))
    i = 0

    for iwl, wl in enumerate(wl_list):
        # ----------------------------
        # slice and reshape arrays

        # ----------------------------
        # plot polar diagrams
        Ncol = len(sed_list)
        Nrow = len(models)
        fig, axs = plt.subplots(Nrow, Ncol, figsize=(Ncol * 7.3, Nrow * 5), subplot_kw=dict(projection='polar'))
        if Nrow == 1:
            axs = np.expand_dims(axs, axis=0)
        fig.subplots_adjust(top=0.9)
        axss = axs.flatten()
        cmap = plt.cm.Spectral_r  # cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
        vmax = np.around(Ixr[iwl, :, :, zidx, szaidx, vza < vzamax, ...].values.max(), 3)
        for isize, size in enumerate(models):
            for ised, sed in enumerate(sed_list):  # sed_list[0::2] + [sed_list[-1]]):
                print(ised, sed)
                I = Ixr[iwl, ised, isize, zidx, szaidx, vza < vzamax, ...].T
                cax = lp.add_polplot(axs[isize, ised], r, theta, I, title='$R_{rs}$ (' + sed + 'mg/L - ' + size + ')',
                                     nlayers=50, scale=False, cmap=cmap)  # , vmin=0, vmax=vmax)
        lp.label_polplot(axs[0, 0])
        plt.suptitle('$R_{rs}\ (sr^{-1})$ at ' + wl + ' micron, SZA = ' + str(sza[szaidx]) + ' deg', size=22)
        plt.tight_layout(rect=[0.0, 0.0, 0.99, 0.95])
        fig.colorbar(cax, ax=axs.ravel().tolist())

        plt.savefig(opj(odir, 'polar_plot', figfile + '_sza' + str(sza[szaidx]) + '_wl' + wl + 'micron.png'), dpi=200,
                    bbox_inches='tight')
        plt.close()

# ---------------------------------
# plot Stokes parameters
# ---------------------------------

ised = 6
# construct raster dimensions
r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))
Nrow = len(models)
for zidx in [1]:
    for iwl, wl in enumerate(wl_list):
        for szaidx in (1, 3, 5):

            fig, axs = plt.subplots(Nrow, 4, figsize=(24, Nrow * 6), subplot_kw=dict(projection='polar'))

            stokesfig = opj(odir, 'polar_plot',
                            'stokes_sediment_spm' + str(sed_[ised]) + '_sza' +
                            str(sza[szaidx]) + '_wl' + wl + 'micron_'+levels[zidx]+'.png')

            for i, size in enumerate(models):
                isize = i
                I = Ixr[iwl, ised, isize, zidx, szaidx, vza < vzamax, ...].T
                Q = Qxr[iwl, ised, isize, zidx, szaidx, vza < vzamax, ...].T
                U = Uxr[iwl, ised, isize, zidx, szaidx, vza < vzamax, ...].T
                DOP = (Q ** 2 + U ** 2) ** 0.5 / I
                # if i == 0:
                #     minmaxI = [np.min(I), np.max(I)]
                #     minmaxQ = [np.min(Q), np.max(Q)]
                #     minmaxU = [np.min(U), np.max(U)]
                #     minmaxDOP = [0, np.max(DOP)]

                # plot polar diagrams
                cmap = cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
                lp.add_polplot(axs[i, 0], r, theta, I, title='I(model ' + models[isize] + ')', cmap=cmap)
                cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
                lp.add_polplot(axs[i, 1], r, theta, Q, title='Q(model ' + models[isize] + ')', cmap=cmap)
                cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
                lp.add_polplot(axs[i, 2], r, theta, U, title='U(model ' + models[isize] + ')', cmap=cmap)
                cmap = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
                lp.add_polplot(axs[i, 3], r, theta, DOP, title='DOP(model ' + models[isize] + ')',
                               cmap=cmap,
                               colfmt='%0.2f')

            plt.suptitle( 'SPM = ' + str(sed_[ised]) + ' mg/L, SZA = ' +
                            str(sza[szaidx]) + 'deg, wl = ' + wl + 'micron, level = '+levels[zidx], fontdict=lp.font)
            plt.tight_layout(rect=[0.0, 0.0, 0.99, 0.95])
            plt.savefig(stokesfig, dpi=300, bbox_inches='tight')
            plt.close()
zidx = 1

# construct raster dimensions
r, theta = np.meshgrid(sza, np.radians(azi))
Nrow = len(models)
for ised in [0,2,6,10]:
    for iwl, wl in enumerate(wl_list):
        for vzaidx in [27]:

            fig, axs = plt.subplots(Nrow, 4, figsize=(24, Nrow * 6), subplot_kw=dict(projection='polar'))

            stokesfig = opj(odir, 'polar_plot',
                            'stokes_vs_sza_sediment_spm' + str(sed_[ised]) + '_vza' +
                            str(vza[vzaidx]) + '_wl' + wl + 'micron_'+levels[zidx]+'.png')

            for i, size in enumerate(models):
                isize = i
                I = Ixr[iwl, ised, isize, zidx, :, vzaidx, ...].T
                Q = Qxr[iwl, ised, isize, zidx, :, vzaidx, ...].T
                U = Uxr[iwl, ised, isize, zidx, :, vzaidx, ...].T
                DOP = (Q ** 2 + U ** 2) ** 0.5 / I
                # if i == 0:
                #     minmaxI = [np.min(I), np.max(I)]
                #     minmaxQ = [np.min(Q), np.max(Q)]
                #     minmaxU = [np.min(U), np.max(U)]
                #     minmaxDOP = [0, np.max(DOP)]

                # plot polar diagrams
                cmap = cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
                lp.add_polplot(axs[i, 0], r, theta, I, title='I(model ' + models[isize] + ')', cmap=cmap)
                cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
                lp.add_polplot(axs[i, 1], r, theta, Q, title='Q(model ' + models[isize] + ')', cmap=cmap)
                cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
                lp.add_polplot(axs[i, 2], r, theta, U, title='U(model ' + models[isize] + ')', cmap=cmap)
                cmap = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
                lp.add_polplot(axs[i, 3], r, theta, DOP, title='DOP(model ' + models[isize] + ')',
                               cmap=cmap,
                               colfmt='%0.2f')

            plt.suptitle( 'SPM = ' + str(sed_[ised]) + ' mg/L, SZA = ' +
                            str(sza[szaidx]) + 'deg, wl = ' + wl + 'micron, level = '+levels[zidx], fontdict=lp.font)
            plt.tight_layout(rect=[0.0, 0.0, 0.99, 0.95])
            plt.savefig(stokesfig, dpi=300, bbox_inches='tight')
            plt.close()
