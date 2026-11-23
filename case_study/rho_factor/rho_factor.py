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

# TODO check sunglint component from OSOAA check OSOAA_SOS_CORE_harmel.F line 2181 or 3982

# ----------------------------

#sys.path.extend(['/home/harmel/Dropbox/work/VRTC/OSOAA_profile/RTxploitation'])
from RTxploitation import lutplot
from RTxploitation import utils as RTu

# the original boa_lut_opac_wind_updown.nc has been superseded by its v2
file = '/data/vrtc/xlut/boa_lut_opac_wind_updown_v2.nc'
xlut = xr.open_dataset(file,chunks={"model":1,"wind":1,"aot_ref":1})
xlut = xlut.drop_vars('z', errors='ignore')
# ----------------------------
# rho factor computation
# ----------------------------
Id = xlut.I.sel(vza=slice(90.001,180.001))
Iu = xlut.I.sel(vza=slice(0,90))
Id['vza']=(180-Id['vza'])
Id=Id.sortby('vza')
Id['vza']=Iu['vza']
rho_factor = Iu/Id
# rho_factor.to_netcdf('/data/vrtc/xlut/rho_factor_osoaa_opac.nc')
# ----------------------------
# plot rho factor for given SZA
# ----------------------------
# construct raster dimensions
lp = lutplot.plot()
vzamax = 61
vza=Iu.vza.values
azi=Iu.azi.values
r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))
iaot=1
szaidx=3
iwl,isza,imodel,iwind,iaot = 2,22,0,2,4
Iu_ = Iu.sel(vza=slice(0,vzamax)).isel(wl=iwl,sza=isza, model=imodel,wind=iwind,aot_ref=iaot)
Id_ = Id.sel(vza=slice(0,vzamax)).isel(wl=iwl,sza=isza, model=imodel,wind=iwind,aot_ref=iaot)
rho_factor_=rho_factor.sel(vza=slice(0,vzamax)).isel(wl=iwl,sza=isza, model=imodel,wind=iwind,aot_ref=iaot)
fig, axs = plt.subplots(1,3, figsize=(20, 8), subplot_kw=dict(projection='polar'))
fig.subplots_adjust(bottom=0.1, top=0.96, left=0.06, right=0.92, hspace=0.12, wspace=0.3)
cmap = cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
lp.add_polplot(axs[ 0], r, theta, Iu_.T, title='I_up', cmap=cmap)
lp.add_polplot(axs[ 1], r, theta, Id_.T, title='I_down', cmap=cmap)
cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
lp.add_polplot(axs[ 2], r, theta, rho_factor_.T, title='rho-factor', cmap=cmap)


# test plots
ivza=27
rho_factor.isel( model=imodel,sza=isza, azi=18, vza=ivza).plot(hue='aot_ref',col='wind',x='wl')



fig, axs = plt.subplots(1,2, figsize=(15, 6))
Id.isel( sza=2, azi=18, vza=vzaidx).plot.line(ax=axs[0],x='wl',marker='o')
Iu.isel( sza=2, azi=18, vza=vzaidx).plot.line(ax=axs[1],x='wl',marker='o')


fig, axs = plt.subplots(1,2, figsize=(15, 6))

rho_factor.isel( sza=6, azi=18, vza=vzaidx).plot.line(ax=axs[1],x='wl',marker='o')

# ----------------------------
# plot rho factor for given SZA
# ----------------------------
# construct raster dimensions
lp = lutplot.plot()
vzamax = 61
vza=Iu.vza.values
azi=Iu.azi.values
r, theta = np.meshgrid(vza[vza < vzamax], np.radians(azi))
iaot=1
szaidx=3
Iu_ = Iu.sel(vza=slice(0,vzamax)).isel(wl=0,sza=6, model=0,wind=0,aot_ref=1)
Id_ = Id.sel(vza=slice(0,vzamax)).isel(wl=0,sza=6, model=0,wind=0,aot_ref=1)
rho_factor_=rho_factor.sel(vza=slice(0,vzamax)).isel(wl=0,sza=6, model=0,wind=0,aot_ref=1)
fig, axs = plt.subplots(1,3, figsize=(20, 8), subplot_kw=dict(projection='polar'))

cmap = cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
lp.add_polplot(axs[ 0], r, theta, Iu_.T, title='I_up', cmap=cmap)
lp.add_polplot(axs[ 1], r, theta, Id_.T, title='I_down', cmap=cmap)
cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
lp.add_polplot(axs[ 2], r, theta, rho_factor_.T, title='rho-factor', cmap=cmap)

vmax=0.06
for i, iwl in enumerate((0,2,4,9)):
    rho = rho_mat[0,0][iaot,iwl, szaidx, vza < vzamax, ...].T
    cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)

    cax = lp.add_polplot(axs[ i], r, theta, rho, title=r'$\rho($' + str(wl_num[iwl]*1000) + ')',
                   cmap=cmap,scale=False, vmin=0,vmax=vmax)
plt.suptitle('SZA = ' + str(sza[szaidx]) + ' deg; AOT = '+aot_list[iaot], **lp.font)
plt.tight_layout()
fig.subplots_adjust(top=.85)
cb = fig.colorbar(cax, ax=axs.ravel().tolist(), shrink=0.6, location='top',pad=0.25)
cb.ax.tick_params(labelsize=19)

plt.savefig(os.path.join(odir,'polar_diag_rho_SZA' + str(sza[szaidx]) + '_AOT'+aot_list[iaot]+'.png'), dpi=300, bbox_inches='tight')
plt.close()


# ----------------------------
# plot rho factor for given VZA
# ----------------------------
# construct raster dimensions
rho_int=rho_factor.interp(sza=np.linspace(0, 80, 100), method='cubic')
sza_int=rho_int.sza.values
r, theta = np.meshgrid(sza_int, np.radians(azi))
iaot=3
vzaidx=27
fig, axs = plt.subplots(1,4, figsize=(20, 8), subplot_kw=dict(projection='polar'))
vmax=0.05
cmap = cm.tools.crop_by_percent(cm.cm.balance, 2, which='both', N=None)
for i, iwl in enumerate((0,2,3,7)):
    rho = rho_int[iaot,iwl, :, vzaidx, ...].T
    cax = lp.add_polplot(axs[ i], r, theta, rho, title=r'$\rho($' + str(wl_num[iwl]*1000) + ')',
                   cmap=cmap,scale=False, vmin=0,vmax=vmax)

plt.suptitle('VZA = ' + str(round(vza[vzaidx])) + ' deg; AOT = '+aot_list[iaot], **lp.font)
plt.tight_layout()
fig.subplots_adjust(top=.85)
cb = fig.colorbar(cax, ax=axs.ravel().tolist(), shrink=0.6, location='top',pad=0.25)
cb.ax.tick_params(labelsize=19)

plt.savefig(os.path.join(odir,'polar_diag_rho_VZA' + str(round(vza[vzaidx])) + '_AOT'+aot_list[iaot]+'.png'), dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------
# plot rho matrix for given VZA
# ----------------------------
# construct raster dimensions
rho_int=rho_factor #.interp(sza=np.linspace(0, 80, 100), method='cubic')
sza_int=rho_int.sza.values
r, theta = np.meshgrid(sza_int, np.radians(azi))
iaot=3
vzaidx=27
iwl=2
fig, axs = plt.subplots(1,4, figsize=(20, 8), subplot_kw=dict(projection='polar'))
vmax=0.5
cmap = cm.tools.crop_by_percent(cm.cm.balance, 2, which='both', N=None)
for i, imat in enumerate((0,1,2)):
    rho = rho_mat[imat,imat][iaot,iwl, :, vzaidx, ...].T
    cax = lp.add_polplot(axs[ i], r, theta, rho, title=r'$\rho($' + str(wl_num[iwl]*1000) + ')',
                   cmap=cmap,scale=False , vmin=-vmax,vmax=vmax)

plt.suptitle('VZA = ' + str(round(vza[vzaidx])) + ' deg; AOT = '+aot_list[iaot], **lp.font)
plt.tight_layout()
fig.subplots_adjust(top=.85)
cb = fig.colorbar(cax, ax=axs.ravel().tolist(), shrink=0.6, location='top',pad=0.25)
cb.ax.tick_params(labelsize=19)

plt.savefig(os.path.join(odir,'polar_diag_rho_VZA' + str(round(vza[vzaidx])) + '_AOT'+aot_list[iaot]+'.png'), dpi=300, bbox_inches='tight')
plt.close()




fig, axs = plt.subplots(2, 4, figsize=(24, 13), subplot_kw=dict(projection='polar'))
if Nrow == 1:
    axs = np.expand_dims(axs, axis=0)
fig.subplots_adjust(top=0.9)

# ----------------------------
# slice and reshape arrays
I = Idf[zidx, szaidx, vza < vzamax, ...].T
Q = Qdf[zidx, szaidx, vza < vzamax, ...].T
U = Udf[zidx, szaidx, vza < vzamax, ...].T
DOP = (Q ** 2 + U ** 2) ** 0.5 / I

# ----------------------------
# plot polar diagrams
cmap = cm.tools.crop_by_percent(cm.cm.delta, 10, which='min', N=None)
lp.add_polplot(axs[i, 0], r, theta, Iu, title='I', cmap=cmap)

cmap = cm.tools.crop_by_percent(cm.cm.balance, 20, which='both', N=None)
lp.add_polplot(axs[i, 1], r, theta, Q, title='Q(' + direction + ')', cmap=cmap)
cmap = cm.tools.crop_by_percent(cm.cm.balance, 1, which='both', N=None)
lp.add_polplot(axs[i, 2], r, theta, U, title='U(' + direction + ')', cmap=cmap)
cmap = cm.tools.crop_by_percent(cm.cm.oxy, 1, which='min', N=None)
lp.add_polplot(axs[i, 3], r, theta, DOP, title='DOP(' + direction + ')', cmap=cmap, colfmt='%0.2f')

plt.suptitle(os.path.basename(file) + ' at SZA = ' + str(sza[szaidx]) + ' deg', **lp.font)
plt.tight_layout()
plt.savefig(figfile + '.png', dpi=300, bbox_inches='tight')
plt.close()


# polarization
rho_mat = np.empty((3,3),dtype=object)
rho_factor = Iuxr.isel(z=0) / Idxr.isel(z=0)
rho_mat[0,0] = rho_factor
rho_mat[0,1] = Iuxr.isel(z=0) / Qdxr.isel(z=0)
rho_mat[0,2] = Iuxr.isel(z=0) / Udxr.isel(z=0)
rho_mat[1,0] = Quxr.isel(z=0) / Idxr.isel(z=0)
rho_mat[1,1] = Quxr.isel(z=0) / Qdxr.isel(z=0)
rho_mat[1,2] = Quxr.isel(z=0) / Udxr.isel(z=0)
rho_mat[2,0] = Uuxr.isel(z=0) / Idxr.isel(z=0)
rho_mat[2,1] = Uuxr.isel(z=0) / Qdxr.isel(z=0)
rho_mat[2,2] = Uuxr.isel(z=0) / Udxr.isel(z=0)

