"""
Particle size distributions and related geometrical parameters
(mean radius, effective radius and variance).
"""
import os

import numpy as np
import pandas as pd
import xarray as xr


# numpy >= 2.0 renamed trapz into trapezoid
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


class psd():
    '''
    Particle size distribution (PSD) functions and conversions between
    the parameters of lognormal distributions.

    Radii ``r`` are given in µm. In the lognormal conversions, ``sig`` is the
    standard deviation of ``ln(r)``.
    '''
    def __init__(self):
        pass

    def norm(self, r, mu=1, sigma=0.5):
        '''
        Normal (Gaussian) distribution.

        :param r: radius
        :param mu: mean radius
        :param sigma: standard deviation
        :return: normalized size distribution at ``r``
        '''
        return (np.exp(-(r - mu) ** 2 / (2 * sigma ** 2))
                / (sigma * np.sqrt(2 * np.pi)))

    def lognorm(self, r, rn_med=1, sigma=0.5):
        '''
        Lognormal number size distribution.

        :param r: radius
        :param rn_med: number median radius
        :param sigma: standard deviation of ``ln(r)``
        :return: normalized size distribution at ``r``
        '''
        mu = np.log(rn_med)
        return (np.exp(-(np.log(r) - mu) ** 2 / (2 * sigma ** 2))
                / (r * sigma * np.sqrt(2 * np.pi)))

    def microplastic_AF2021(self, r, gamma=.5):
        '''
        Size distribution for Microplastics from Aoki and Furue, 2021,
        https://doi.org/10.1371/journal.pone.0259781

        :param r: radius (diameter is ``d = 2 * r``)
        :param gamma: shape parameter of the distribution
        :return: size distribution normalized to unity over ``r``
        '''
        d = 2 * r
        psd = d ** -4 / (np.exp(1 / (gamma * d)) - 1)
        norm = _trapezoid(psd, r)
        return psd/norm

    def power_law_junge(self, r, slope=-3.5, rmin=0.03, rmax=100):
        '''
        Junge (power-law) size distribution ``r ** slope``, set to zero
        outside ``[rmin, rmax]``.

        :param r: radius (numpy array)
        :param slope: power-law exponent
        :param rmin: minimum radius
        :param rmax: maximum radius
        :return: size distribution normalized to unity over ``r``
        '''
        psd = np.array(r ** slope)
        psd[r < rmin] = 0.
        psd[r > rmax] = 0.
        return psd / _trapezoid(psd, r)

    def modif_power_law(self, r, slope=-3.5, rmin=0.03, rmax=100):
        '''
        Modified power-law size distribution ``(r / rmin) ** slope``,
        constant below ``rmin`` and set to zero above ``rmax``.

        :param r: radius (numpy array)
        :param slope: power-law exponent
        :param rmin: radius below which the distribution is constant
        :param rmax: maximum radius
        :return: size distribution normalized to unity over ``r``
        '''
        psd = np.array((r / rmin) ** slope)
        psd[r < rmin] = 1
        psd[r > rmax] = 0.
        return psd / _trapezoid(psd, r)

    def rmod2rmed(self, rmod, sig):
        '''
        Convert the mode radius of a lognormal distribution into its
        median radius.

        :param rmod: mode radius
        :param sig: standard deviation of ``ln(r)``
        :return: median radius
        '''
        return np.exp(np.log(rmod) + sig ** 2)

    def rmed2rmod(self, rmed, sig):
        '''
        Convert the median radius of a lognormal distribution into its
        mode radius.

        :param rmed: median radius
        :param sig: standard deviation of ``ln(r)``
        :return: mode radius
        '''
        return np.exp(np.log(rmed) - sig ** 2)

    def muv2mun(self, muv, sig):
        '''
        Convert the mean of ``ln(r)`` of the volume distribution into that
        of the number distribution.

        :param muv: mean of ``ln(r)`` for the volume distribution
        :param sig: standard deviation of ``ln(r)``
        :return: mean of ``ln(r)`` for the number distribution
        '''
        return muv - 3 * sig ** 2

    def mun2muv(self, mun, sig):
        '''
        Convert the mean of ``ln(r)`` of the number distribution into that
        of the volume distribution.

        :param mun: mean of ``ln(r)`` for the number distribution
        :param sig: standard deviation of ``ln(r)``
        :return: mean of ``ln(r)`` for the volume distribution
        '''
        return mun + 3 * sig ** 2

    def rvmod2rnmed(self, rv_mod, sig):
        '''
        Convert the volume mode radius into the number median radius.

        :param rv_mod: mode radius of the volume distribution
        :param sig: standard deviation of ``ln(r)``
        :return: median radius of the number distribution
        '''
        return rv_mod * np.exp(-2 * sig ** 2)

    def rnmed2rvmed(self, rn_med, sig):
        '''
        Convert the number median radius into the volume median radius.

        :param rn_med: median radius of the number distribution
        :param sig: standard deviation of ``ln(r)``
        :return: median radius of the volume distribution
        '''
        return np.exp(np.log(rn_med) + 3 * sig ** 2)

    def arr_format(self, arr, fmt="{:0.1f}"):
        '''
        Format numbers into strings.

        :param arr: iterable of numbers
        :param fmt: format string
        :return: list of formatted strings
        '''
        return [fmt.format(x) for x in arr]


class size_param:
    '''
    Size Distribution and Geometrical Parameters
    (see Hansen and Travis, 1974).

    Note that ``psd`` is modified in place: NaN values are set to 0 for the
    computation, then zero values are set to NaN.

    :param radius: radius (µm)
    :param psd: number size distribution at ``radius``

    :ivar rmean: mean radius
    :ivar G: mean geometrical cross section
    :ivar reff: effective radius
    :ivar veff: effective variance (dimensionless)
    :ivar sigeff: square root of ``veff``
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
        '''
        :return: dictionary of the geometrical parameters
        '''
        return {
            'G': self.G,
            'rmean': self.rmean,
            'reff': self.reff,
            'veff': self.veff,
            'sigeff': self.sigeff,
        }

    def to_annotation_r(self):
        '''
        :return: string (LaTeX) of mean and effective radius for plot
            annotation
        '''
        return '$r_{mean}=$' + '{:.2e}'.format(self.rmean) + r' $\mu m$' + '\n $r_{eff}=$' + '{:.2e}'.format(
            self.reff) + r' $\mu m$'

    def to_annotation(self):
        '''
        :return: one-line string (LaTeX) of mean radius, effective radius
            and effective variance for plot annotation
        '''
        return '$r_{mean}=$' + '{:.3f}'.format(self.rmean) + r'$\mu m,\ r_{eff}=$' + '{:.3f}'.format(
            self.reff) + r'$\mu m,\ \sigma^2_{eff}=$' + '{:.3f}'.format(self.veff) + r'$\mu m^2$'

    def integr(self, order=1):
        '''
        Integral of ``psd * radius ** order`` over radius.

        :param order: power of the radius
        :return: integral value
        '''
        _f = self.psd * self.radius ** order
        return _trapezoid(_f, self.radius)

    def moment(self, order=1):
        '''
        Integral of ``pi * psd * radius ** order`` over radius.

        :param order: power of the radius
        :return: integral value
        '''
        _f = self.psd * np.pi * self.radius ** order
        return _trapezoid(_f, self.radius)

    def variance(self, G, reff):
        '''
        Effective variance of the size distribution.

        :param G: mean geometrical cross section
        :param reff: effective radius
        :return: effective variance
        '''
        _f = self.psd * np.pi * ((self.radius - reff) * self.radius) ** 2
        return _trapezoid(_f, self.radius) / (reff ** 2 * G)
