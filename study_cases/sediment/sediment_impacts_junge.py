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
plt.ioff()
plt.rcParams.update({'font.size': 16})

# TODO check sunglint component from OSOAA check OSOAA_SOS_CORE_harmel.F line 2181 or 3982

# ----------------------------
# TODO put as package
#sys.path.extend(['/home/harmel/Dropbox/work/VRTC/OSOAA_profile/RTxploitation'])
from RTxploitation import lutplot
from RTxploitation import utils as u

idir = '/DATA/projet/VRTC/lut/bagré'
odir = '/DATA/projet/bagre/fig/'

wl_num = [0.665, 0.865]
wl_list = u.misc.arr_format(wl_num, "{:0.3f}")
sed_list = u.misc.arr_format([2.00, 10., 20., 50., 100., 150., 200., 300., 400., 600.], "{:0.2f}")
sedjs_list = u.misc.arr_format([3.05, 3.5, 3.7, 4, 5], "{:0.2f}")
sedni_list = u.misc.arr_format([-1e-4, -1e-3, -1e-2], "{:0.4f}")
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
                        dims=('wl', 'sed', 'sedjs', 'z', 'sza', 'vza', 'azi'),
                        coords={'wl': wl_num, 'sed': s2f(sed_list), 'sedjs': s2f(sedjs_list), 'z': z, 'sza': sza,
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
sedni = sedni_list[0]
for sedni in sedni_list:

    figfile = 'Rrs_sednr1.17_sedni' + sedni

    first = True

    for iwl, wl in enumerate(wl_list):
        for ised, sed in enumerate(sed_list):
            for isedjs, sedjs in enumerate(sedjs_list):

                pattern = 'osoaa_tot_aot0.1_aero_rg0.80_sig0.60_nr1.45_ni-0.0010_ws2_chl3.00_sed' + sed + '_sedjs' + sedjs + '_sednr1.17_sedni' + sedni + '_wl' + wl + '.nc'

                file = os.path.join(idir, pattern)

                lut = nc.Dataset(file)

                sed_prop = lut['/optical_properties/sediment'].variables
                keys = list(sed_prop.keys())
                ds = pd.DataFrame(sed_prop['F11'][:], columns=['vsf'], index=sed_prop['scatt_ang'][:])
                ds['wl'] = float(wl)
                ds['sed'] = float(sed)
                ds['sedjs'] = float(sedjs)
                ds['sedni'] = float(sedni)
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
                    dim = (len(wl_list), len(sed_list), len(sedjs_list)) + Nlut
                    Rrs,I, Q, U = u.misc.init_array(4, dim)
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
                I[iwl, ised, isedjs, ...] = stokes.variables['I'][z_slice:]/np.pi
                Q[iwl, ised, isedjs, ...] = stokes.variables['Q'][z_slice:]/np.pi
                U[iwl, ised, isedjs, ...] = stokes.variables['U'][z_slice:]/np.pi

                for isza,sza_ in  enumerate(sza):
                    print(sza_)
                    Rrs[:,:,:,:,isza, ...]=I[:,:,:,:,isza, ...]/np.cos(np.pi/180*sza_)


    # ---------------------------------
    # plot vsf/optical data
    # ---------------------------------
    colors = ["pale red", "amber", "greyish", "faded green", "dusty purple"] * 4

    sed_props.sort_values(by='sedni', inplace=True)
    sed_props.sort_index(inplace=True)
    sed_props['legend'] = '$n_i = $' + sed_props.sedni.apply(lambda x: '{0:.0e}'.format(-x)) + '$; g = $' + \
                          sed_props.asym_factor.apply(lambda x: '{0:.3f}'.format(x)) + '$; b_{bp}/b_p = $' + \
                          sed_props.bbp_ratio.apply(lambda x: '{0:.4f}'.format(x)) + '$; ssa = $' + \
                          sed_props.ssa.apply(lambda x: '{0:.2f}'.format(x))+ '$; r_{eff} = $' + \
                          sed_props.reff.apply(lambda x: '{0:.3f}'.format(x))
    sed_props_sub = sed_props[sed_props.sed == float(sed_list[0])]

    import seaborn as sns
    # ---------------------------------
    # plot VSF
    # ---------------------------------
    sns.set(style="ticks", color_codes=True)
    g = sns.FacetGrid(sed_props_sub, col="wl", row="sedjs", hue="legend", palette='tab20c')  # sns.xkcd_palette(colors))

    g.map(plt.semilogy, 'scatt_ang', 'vsf', linewidth=2)
    for ax in g.axes.ravel():
        h, l = ax.get_legend_handles_labels()
        neworder = sorted(range(len(l)), key=l.__getitem__)
        l = [l[i] for i in neworder]
        h = [h[i] for i in neworder]

        ax.legend(h, l)
    g.fig.set_size_inches(14,16)
    g.savefig(opj(odir,'vsf','VSF_used.pdf'))


    # ---------------------------------
    # plot Rrs data
    # ---------------------------------
    Ixr, Qxr, Uxr = _toxr(Rrs), _toxr(Q), _toxr(U)

    # one direction plots
    zidx = 0
    vzaidx = 27
    aziidx = 18

    for szaidx in (1, 3, 5):
        sza_ = str(sza[szaidx])
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        Ixr.isel(sza=szaidx, azi=aziidx, vza=vzaidx).plot.line(x='sed', marker='o', row='z', hue='sedjs', col='wl')
        plt.suptitle('SZA = ' + sza_ + ' deg, VZA = ' + str(vza[vzaidx]) + ' deg, AZI = ' + str(azi[aziidx]),
                     fontdict=lp.font)
        plt.tight_layout(rect=[0.0, 0.0, 0.99, 0.9])
        plt.savefig(opj(odir,'onedir',figfile + '_sza'+sza_+'_onedir.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ----------------------------

        # construct raster dimensions
        r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))
        i = 0

        for iwl, wl in enumerate(wl_list):
            # ----------------------------
            # slice and reshape arrays
            I = Ixr[0, 0, 0, zidx, szaidx, vza < vzamax, ...].T
            # Q = Qdf[zidx, szaidx, vza < vzamax, ...].T
            # U = Udf[zidx, szaidx, vza < vzamax, ...].T
            # DOP = (Q ** 2 + U ** 2) ** 0.5 / I

            # ----------------------------
            # plot polar diagrams

            Nrow = len(sedjs_list)
            fig, axs = plt.subplots(Nrow, 6, figsize=(44, Nrow*5), subplot_kw=dict(projection='polar'))
            if Nrow == 1:
                axs = np.expand_dims(axs, axis=0)
            fig.subplots_adjust(top=0.9)
            axss = axs.flatten()
            cmap = plt.cm.Spectral_r #cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
            vmax = np.around(Ixr[iwl, :, :, zidx, szaidx, vza < vzamax, ...].values.max(), 3)
            for ijs, js in enumerate(sedjs_list):
                for ised, sed in enumerate(sed_list[0::2] + [sed_list[-1]]):
                    print(ised, sed)
                    I = Ixr[iwl, ised, ijs, zidx, szaidx, vza < vzamax, ...].T
                    cax = lp.add_polplot(axs[ijs, ised], r, theta, I, title='I (' + sed + 'mg/L - JS = ' + js + ')',
                                         nlayers=50, scale=False, cmap=cmap)#, vmin=0, vmax=vmax)
            lp.label_polplot(axs[0,0])
            plt.suptitle('$R_{rs}\ (sr^{-1})$ at '+wl+' micron, SZA = '+ str(sza[szaidx]) + ' deg', size=22)
            plt.tight_layout(rect=[0.0, 0.0, 0.99, 0.95])
            fig.colorbar(cax, ax=axs.ravel().tolist())

            plt.savefig(opj(odir,'polar_plot',figfile + '_sza' + str(sza[szaidx]) +'_wl' + wl + 'micron.png'), dpi=200, bbox_inches='tight')
            plt.close()

        # cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
        # lp.add_polplot(axs[i, 1], r, theta, Q, title='Q(' + direction + ')', cmap=cmap)
        # cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
        # lp.add_polplot(axs[i, 2], r, theta, U, title='U(' + direction + ')', cmap=cmap)
        # cmap = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
        # lp.add_polplot(axs[i, 3], r, theta, DOP, title='DOP(' + direction + ')', cmap=cmap, colfmt='%0.2f')
