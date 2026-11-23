import os, copy
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ioff()
plt.rcParams.update({'font.size': 16})

from Py6S import *

opj = os.path.join
dir = os.path.abspath('/DATA/projet/garaba/boa_to_toa')
file = opj(dir, 'Harmel_Sample.xlsx')

df = pd.read_excel(file, skiprows=2, index_col=0)
wl = df.index
sza, vza, azi = 30, 5, 90
aot=0.7
aerosols = [['maritime',AeroProfile.Maritime],
            ['continental',AeroProfile.Continental],
            ['desert',AeroProfile.Desert]]
aerosol=aerosols[1]
amodel=aerosol[0]
figfile='plastic_impact_boa_toa_amodel_'+amodel+'_aot'+str(aot)+'_sza'+str(sza)+'_vza'+str(vza)+'_azi'+str(azi)+'.png'

####################################
#     6S absorbing gases transmittance
####################################
N = len(wl)
s = SixS()
s.geometry.solar_z = sza
s.geometry.solar_a = 0
s.geometry.view_z = vza
s.geometry.view_a = azi
s.altitudes.set_sensor_satellite_level()
s.aot550 = aot
s.aero_profile = AeroProfile.PredefinedType(aerosol[1])


# wavelengths, results = SixSHelpers.Wavelengths.run_wavelengths(s, wl / 1000)

def proc(p):
    wl, ground_reflec = p
    wl = wl / 1000  # convert nm -> microns
    print(wl)
    s.outputs = None
    a = copy.deepcopy(s)
    a.wavelength = Wavelength(wl)
    a.ground_reflectance = GroundReflectance.HomogeneousLambertian(ground_reflec)
    a.run()

    return a.outputs


pool = Pool()
refl_spectrum = df.iloc[:, 1].reset_index().__array__()
results = pool.map(proc, refl_spectrum)
pool.close()
pool.join()

F0, trans_gas, trans_scat, irradiance, = [], [], [], []
toa_refl, intrinsic_refl, intrinsic_rad = [], [], []

for res in results:
    print(res.atmospheric_intrinsic_reflectance)
    F0 = np.append(F0, res.solar_spectrum)
    trans_scat = np.append(trans_scat, res.transmittance_total_scattering.total)
    trans_gas = np.append(trans_gas, res.transmittance_global_gas.total)
    irradiance = np.append(irradiance,
                           res.diffuse_solar_irradiance + res.direct_solar_irradiance)
    toa_refl = np.append(toa_refl, res.pixel_reflectance)
    intrinsic_refl = np.append(intrinsic_refl, res.atmospheric_intrinsic_reflectance)
    intrinsic_rad = np.append(intrinsic_rad, res.atmospheric_intrinsic_radiance)

Es_toa = F0 * np.cos(np.radians(sza))
trans_tot = trans_gas * trans_scat

# ------------------------------
# Plotting section
# ------------------------------

cmap = plt.cm.nipy_spectral
norm = mpl.colors.Normalize(vmin=0, vmax=100)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
nrows = 3
fig, axs = plt.subplots(nrows=nrows, figsize=(12, 15))
axs = axs.ravel()
axs[0].plot(wl, trans_tot, '--k')
for pixcov, group in df.iteritems():
    print(pixcov)

    y = group.values
    axs[0].plot(wl, y, color=cmap(norm(pixcov)), lw=1.5, markersize=2, alpha=0.5)
    axs[1].plot(wl, trans_tot * y+intrinsic_refl, '--', color=cmap(norm(pixcov)), lw=1.5, markersize=2, alpha=0.5)
    axs[2].plot(wl, Es_toa/np.pi * trans_tot * y+intrinsic_rad,'--', color=cmap(norm(pixcov)), lw=1.5, markersize=2, alpha=0.5)

for i in range(nrows):
    divider = make_axes_locatable(axs[i])
    cax = divider.append_axes('right', size='2%', pad=0.05)
    cbar = fig.colorbar(sm, cax=cax, format=mpl.ticker.ScalarFormatter(),
                        shrink=1.0, fraction=0.1, pad=0)
    axs[i].set_xlabel('Wavelength (nm)')

axs[0].set_ylabel('BOA Reflectance $(-)$')
axs[1].set_ylabel('TOA Reflectance $(-)$')
axs[2].set_ylabel('TOA Radiance\n $(W\ m^{-2}\ sr^{-1}\ \mu m^{-1})$')
plt.tight_layout()
plt.savefig(opj(dir,'fig',figfile),dpi=200)