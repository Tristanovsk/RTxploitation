import os

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
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

from RTxploitation import utils as u
import RTxploitation.load_lut_nc as load
from RTxploitation import lutplot

opj = os.path.join
lp = lutplot.plot()

odir = "/DATA/projet/garaba/OP3/fig/"
figdir='./study_cases/transmittance/fig'
plot = True

# load Rayleigh optical thickness
rot = pd.read_csv('./data/rayleigh_bodhaine.txt',sep='\s+',skiprows=16,header=None)
rot.columns=['wl','rot','dpol']
xrot = rot.set_index('wl').to_xarray()
xrot=xrot.sel(wl=slice(400,850))

aot550=0.1
ang_exp = 1.1
aot=aot550*(xrot.wl/550.)**-ang_exp

def arr_format(arr, fmt="{:0.3f}"):
    return [fmt.format(x) for x in arr]

# ---------------------------------
# Load and arrange data
# ---------------------------------
labels=[]
spm_norm='10.00'
nr='1.2'
rmed='10.0'
wl_= np.arange(400,851,25)/1e3
wls = arr_format(wl_, "{:0.3f}")
for wl in wls:
    labels.append('__sed%s_nr%s_rmed%s_wl%s'%(spm_norm,nr,rmed,wl))
xStokes = load.load_osoaa(labels=labels,water_signal=True)


z=0
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
fig.subplots_adjust(bottom=0.115, top=0.98, left=0.086, right=0.98,
                    hspace=0.1, wspace=0.05)
xStokes.I.isel(z=z,vza=6,sza=3,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[0])
xStokes.Q.isel(z=z,vza=6,sza=3,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[1])
xStokes.U.isel(z=z,vza=6,sza=3,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[2])
plt.show()

Trans = xStokes.isel(z=-1)/xStokes.isel(z=0)
ivza=8
szas=Trans.sza.values
vzas=Trans.vza.values

muv = np.cos(np.radians(Trans.vza))
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15),sharex=True,sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.94, left=0.086, right=0.9,
                    hspace=0.1, wspace=0.05)

for isza,sza in enumerate([1,3,6]):

    mu0= np.cos(np.radians(szas[sza]))
    td=np.exp(-(0.52*xrot.rot+0.14*aot)/muv.isel(vza=ivza))
    td_E=np.exp(-(0.52*xrot.rot+0.16*aot)/mu0)
    Trans.I.isel(vza=ivza,sza=sza,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,0],add_legend=False)
    axs[isza,0].plot(xrot.wl,td,color='black')
    axs[isza,0].plot(xrot.wl,td_E,color='red')
    axs[isza,0].plot(xrot.wl,td*td_E,'--',color='black')

    Trans.Q.isel(vza=ivza,sza=sza,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,1],add_legend=False)
    Trans.U.isel(vza=ivza,sza=sza,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[isza,2],add_legend=False)
plt.show()


fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12),sharex=True,sharey=True)
fig.subplots_adjust(bottom=0.1, top=0.94, left=0.086, right=0.96,
                    hspace=0.15, wspace=0.1)
for iv,ivza in enumerate([1,5,11]):
    for i,isza in enumerate([1,3,6]):

        np.exp(-(0.52*xrot.rot+0.16*aot) / muv.isel(vza=ivza)).plot(x='wl',hue='azi',ax=axs[iv,i],add_legend=False)
        Trans.I.isel(vza=ivza,sza=isza,azi=[0,10,20,30,40]).plot(x='wavelength',hue='azi',ax=axs[iv,i],add_legend=False)
        axs[iv, i].set_title("$VZA="+str(vzas[ivza])+",\ SZA="+str(szas[isza])+"$",fontsize=16)
        axs[iv, i].set(xlabel=None,ylabel=None)
for i in range(3):
    axs[-1, i].set_xlabel('$Wavelength\ (nm)$')
    axs[i,0].set_ylabel('$Total\ transmittance$')
plt.savefig(opj(figdir, 'upward_transmittance_test.png'), dpi=300)
plt.show()
