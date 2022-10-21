import os

import numpy as np
import pandas as pd
import xarray as xr


class psd():
    def __init__(self):
        pass

    def norm(self, r, mu=1, sigma=0.5):
        return (np.exp(-(r - mu) ** 2 / (2 * sigma ** 2))
                / (sigma * np.sqrt(2 * np.pi)))

    def lognorm(self, r, rn_med=1, sigma=0.5):
        mu = np.log(rn_med)
        return (np.exp(-(np.log(r) - mu) ** 2 / (2 * sigma ** 2))
                / (r * sigma * np.sqrt(2 * np.pi)))

    def microplastic_AF2021(self, r, gamma=.5):
        '''
        Size distribution for Microplastics from Aoki and Furue, 2021,
        https://doi.org/10.1371/journal.pone.0259781
        :return:
        '''
        d = 2 * r
        psd = d ** -4 / (np.exp(1 / (gamma * d)) - 1)
        norm = np.trapz(psd, r)
        return psd/norm

    def power_law_junge(self, r, slope=-3.5, rmin=0.03, rmax=100):
        psd = np.array(r ** slope)
        psd[r < rmin] = 0.
        psd[r > rmax] = 0.
        return psd / np.trapz(psd, r)

    def modif_power_law(self, r, slope=-3.5, rmin=0.03, rmax=100):
        psd = np.array((r / rmin) ** slope)
        psd[r < rmin] = 1
        psd[r > rmax] = 0.
        return psd / np.trapz(psd, r)

    def rmod2rmed(self, rmod, sig):
        return np.exp(np.log(rmod) + sig ** 2)

    def rmed2rmod(self, rmed, sig):
        return np.exp(np.log(rmed) - sig ** 2)

    def muv2mun(self, muv, sig):
        return muv - 3 * sig ** 2

    def mun2muv(self, mun, sig):
        return mun + 3 * sig ** 2

    def rvmod2rnmed(self, rv_mod, sig):
        return rv_mod * np.exp(-2 * sig ** 2)

    def rnmed2rvmed(self, rn_med, sig):
        return np.exp(np.log(rn_med) + 3 * sig ** 2)

    def arr_format(self, arr, fmt="{:0.1f}"):
        return [fmt.format(x) for x in arr]


class size_param:
    '''
     Size Distribution and Geometrical Parameters
    '''

    def __init__(self, radius, psd):
        self.psd = psd
        self.psd[np.isnan(self.psd)] = 0
        self.radius = radius
        self.rmean = self.integr() / self.integr(0)
        self.G = self.moment(2)
        self.reff = self.moment(3) / self.G
        self.veff = self.variance(self.G, self.reff)
        self.sigeff = self.veff ** 0.5
        self.psd[self.psd == 0] = np.nan

    def to_dict(self):
        return {
            'G': self.G,
            'rmean': self.rmean,
            'reff': self.reff,
            'veff': self.veff,
            'sigeff': self.sigeff,
        }

    def to_annotation_r(self):
        return '$r_{mean}=$' + '{:.2e}'.format(self.rmean) + ' $\mu m$\n $r_{eff}=$' + '{:.2e}'.format(
            self.reff) + ' $\mu m$'

    def to_annotation(self):
        return '$r_{mean}=$' + '{:.3f}'.format(self.rmean) + '$\mu m,\ r_{eff}=$' + '{:.3f}'.format(
            self.reff) + '$\mu m,\ \sigma^2_{eff}=$' + '{:.3f}'.format(self.veff) + '$\mu m^2$'

    def integr(self, order=1):
        _f = self.psd * self.radius ** order
        return np.trapz(_f, self.radius)

    def moment(self, order=1):
        _f = self.psd * np.pi * self.radius ** order
        return np.trapz(_f, self.radius)

    def variance(self, G, reff):
        _f = self.psd * np.pi * ((self.radius - reff) * self.radius) ** 2
        return np.trapz(_f, self.radius) / (reff ** 2 * G)
