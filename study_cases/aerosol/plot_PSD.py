import os

import numpy as np
import pandas as pd
import xarray as xr
import glob
from scipy.integrate import trapz  # import a single function for integration using trapezoidal rule

import matplotlib.pyplot as plt
import matplotlib as mpl
from RTxploitation.psd import psd, size_param

rc = {"font.family": "serif",
      "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

plt.rcParams.update({'font.size': 18, 'axes.labelsize': 22})

# --------------------------
# Size distrib param
rn_med_f, sigma_f = 0.1, 0.4
rn_med_c, sigma_c = 0.81, 0.6

r = np.logspace(-2, 2, 1000)
# r = np.logspace(-6, 2, 5000)
vol = 4 / 3 * np.pi * r ** 3
psdN_f = psd().lognorm(r, rn_med=rn_med_f, sigma=sigma_f)
psdN_c = psd().lognorm(r, rn_med=rn_med_c, sigma=sigma_c)


def V0_lognorm(rn_med=0.1, sigma=0.4):
    return 4 / 3 * np.pi * np.exp(3 * np.log(rn_med) + 4.5 * sigma * 2)


V0_f = V0_lognorm(rn_med=rn_med_f, sigma=sigma_f)
V0_c = V0_lognorm(rn_med=rn_med_c, sigma=sigma_c)
psdV_f = vol / V0_f * psdN_f
psdV_c = vol / V0_c * psdN_c

#
#
# rv_mod = size_param.rmed2rmod(rv_med, sig)
# rn_med = rvmod2rnmed(rv_mod, sig)
# rn_mod = rmed2rmod(rn_med, sig)
# muv=np.log(rv_med)
# mun=np.log(rn_med)

# ----------------
# plot dN/dr and dV/dlogr
# ----------------
cvfs=[0.025,0.1,0.25,0.5,0.75,0.9]


cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     'gray', 'yellowgreen', 'forestgreen','gold','darkgoldenrod']).reversed()
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig, axs = plt.subplots(2, 1, figsize=(7, 11), sharex=True)
# fig.subplots_adjust(bottom=0.15, top=0.94, left=0.1, right=0.975,hspace=0.05,wspace=0.25)
fig.subplots_adjust(bottom=0.1, top=0.96, left=0.2, right=0.95, hspace=0.05, wspace=0.25)
ax = axs[0]
# ax.plot(r, psdN_f)
# ax.plot(r, psdN_c)

for cvf in cvfs:
    ax.plot(r,r*(cvf/V0_f*psdN_f+(1-cvf)/V0_c*psdN_c),color=cmap(norm(cvf)),label='CVfine{:.3f}'.format(cvf))
    ax.plot(r,r *(cvf* psdV_f+(1-cvf)*psdV_c)/vol,ls='--',color=cmap(norm(cvf)))
ax.loglog()
ax.tick_params(axis='x', which='major', length=7, width=1.2)
ax.tick_params(axis='x', which='minor', length=4)
#ax.set_ylabel(r'$dN(r)/dr\ (\mu m^{-1} \cdot \ cm^{-3})$')
ax.set_ylabel(r'$dN(r)/dlogr\ (cm^{-3})$')
# ax.set_xlabel(r'$Radius\ (\mu m)$')
ax.legend(fontsize=13)
#ax.set_ylim([1e-7, 0.2e2])
ax.text(-0.225, .96, '(a)', transform=ax.transAxes, size=18)  # , weight='bold')

ax = axs[1]
# ax.plot(r, r * psdV_f)
# ax.plot(r, r * psdV_c)
for cvf in cvfs:
    print(cvf,(1-cvf)/cvf)
    ax.plot(r, r *(cvf* psdV_f+(1-cvf)*psdV_c),color=cmap(norm(cvf)),label='CVfine{:.3f}'.format(cvf))
ax.tick_params(axis='x', which='major', length=7, width=1.2)
ax.tick_params(axis='x', which='minor', length=4)
# ax2 = ax.twinx()

ax.set_ylabel(r'$dV(r)/dlogr\ (\mu m^3\cdot \ cm^{-3})$')
ax.semilogx()

ax.set_xlabel(r'$Radius\ (\mu m)$')
ax.set_xlim([0.01, 200])
#ax.set_ylim([0.0, 0.72])
ax.text(-0.225, .96, '(b)', transform=ax.transAxes, size=18)  # , weight='bold')

ax.legend(fontsize=13)
plt.show()

for cvf in cvfs:

    cnf=cvf/V0_f
    cnc=(1-cvf)/V0_c
    N0 = cnc+cnf
    print(cvf,(1-cvf)/cvf,cnf/N0)
CVc_CVf = (1-cvf)/cvf
# plt.savefig('fig/PSD_dN_dr_dV_dlogr_SPM_power_law_'+lut+'.png',dpi=300)
#
