"""
Bio-optical inversion of remote sensing reflectance (version 1).

Inverts above-water Rrs into chlorophyll-a concentration and the absorption
and backscattering of a "background" component (CDOM + non-algal particles)
using the Gordon et al. (1988) model, chlorophyll fluorescence (Gower et al.
2004) and the specific IOPs (a*, bb*) of the dominant algae species
(e.g., *Mesodinium rubrum*).

In this version the cost function is expressed in terms of chlorophyll
concentration (algebraic inversion of the Gordon model for each band).
See ``solver_v2.py`` for the version minimizing the Rrs residuals.
"""
import numpy as np
import xarray as xr
import lmfit as lm

import RTxploitation as RT

RTp = RT.parameterization
ad = RT.auxdata
spectral = RT.spectral_integration.spectral_responses
water = RTp.water()


class Rrs_inversion:
    """
    Inversion of Rrs spectra into chl-a and background IOPs.

    The unknowns are ``chl`` (mg m-3), ``a_bg_ref`` and ``bb_bg_ref``
    (background absorption and backscattering at 440 nm, m-1),
    ``S_bg`` (spectral slope of absorption, nm-1) and ``eta_bg`` (spectral
    slope of backscattering).
    """
    def __init__(self, a_star: xr.DataArray, bb_star: xr.DataArray, sza, sat='s2a',band_idx=None, wl_=None):
        '''
        Set parameter values for satellite bands characteristics (band spectral responses)
        :param sat: satellite sensor ID
        :param a_star: spectral specific absorption coef. for the major algae species
        :param bb_star: spectral  back-scattering coef. for the major algae species
        :param sza: sun zenith angle (deg.)
        '''
        if wl_ is not None:
            # mono-spectral bands
            self.wl = wl_
            self.a_star = a_star.interp(wl=wl_)
            self.bb_star = bb_star.interp(wl=wl_)
            self.aw, self.bbw = ad.iopw().get_iopw(wl_)
        else:
            # proceed to band spectral response convolution
            rsrs = spectral(sat,band_idx)
            wl_ = np.arange(350, 2700, 1)

            self.wl = rsrs.convolution(wl_, wl_)
            self.a_star = rsrs.xr_convolution(a_star)
            self.bb_star = rsrs.xr_convolution(bb_star)
            aw, bbw = ad.iopw().get_iopw(wl_)
            self.aw, self.bbw = rsrs.convolution(wl_, aw), rsrs.convolution(wl_, bbw)


        # parameters for the Gordon88 model
        self.g0 = 0.089
        self.g1 = 0.125
        self.g0sq = self.g0 ** 2

        self.Ed_ref = RTp.atmo().Ed_singlewl(sza, 865)

    def Rrs_to_iops(self, Rrs, wl_nir_threshold=715):
        '''
        Simple semi-analytical retrieval of total absorption and
        backscattering: bbp is estimated from the bands beyond
        ``wl_nir_threshold`` (where absorption is dominated by water) and
        assumed spectrally flat.

        :param Rrs: remote sensing reflectance above water (sr-1)
        :param wl_nir_threshold: lower wavelength (nm) of the bands used for bbp
        :return: total absorption and backscattering coefficients (m-1)
        '''
        rrs = water.Rrs2rrs(Rrs)

        u = water.inv_gordon88(rrs)

        idx = self.wl > wl_nir_threshold
        bbp = self.aw[idx] * u[idx] / (1. - u[idx]) - self.bbw[idx]

        # '_' for retrieved parameters
        self.bbp_ = bbp.mean()
        self.bb_ = self.bbp_ + self.bbw
        self.a_ = self.bb_ / u - self.bb_

        return self.a_, self.bb_

    def abg(self, a_bg_ref, S_bg, wl_ref=440):
        '''
        Exponential absorption spectrum of the background component.

        :param a_bg_ref: absorption at ``wl_ref`` (m-1)
        :param S_bg: spectral slope (nm-1)
        :param wl_ref: reference wavelength (nm)
        :return: absorption at ``self.wl`` (m-1)
        '''
        return a_bg_ref * np.exp(-S_bg * (self.wl - wl_ref))

    def bbbg(self, bb_bg_ref, eta_bg, wl_ref=440):
        '''
        Power-law backscattering spectrum of the background component.

        :param bb_bg_ref: backscattering at ``wl_ref`` (m-1)
        :param eta_bg: spectral slope (-)
        :param wl_ref: reference wavelength (nm)
        :return: backscattering at ``self.wl`` (m-1)
        '''
        return bb_bg_ref * (wl_ref / self.wl) ** eta_bg

    def objfunc(self, x, rrs, a_star, bb_star):
        '''
        Objective function to be minimized
        :param x: vector of unknowns
        :param a_star: bulk absorption coefficient specific to Chl-a for each wl (in m^2 mg^{-1})
        :param bb_star: bulk backscattering coefficient specific to Chl-a for each wl (in m^2 mg^{-1})
        :param sza: solar zenith angle in deg (for fluorescence computation)
        :return: vector of the objective function values for each wl
        '''

        chl, a_bg_ref, bb_bg_ref, S_bg, eta_bg = np.array(list(x.valuesdict().values()))
        # print(chl, a_bg_ref, bb_bg_ref, S_bg, eta_bg)
        a_bg = self.abg(a_bg_ref, S_bg)
        bb_bg = self.bbbg(bb_bg_ref, eta_bg)
        rrs_fluo = water.Rrs2rrs(water.fluo_gower2004(chl, self.wl, Ed_ref=self.Ed_ref))

        deltas = self.g0sq + 4 * self.g1 * (rrs - rrs_fluo)

        u = (-self.g0 + np.sqrt(deltas)) / (2 * self.g1)

        func = chl + (self.bbw + bb_bg - u * (self.aw + a_bg + self.bbw + bb_bg)) \
               / (bb_star - u * (a_star + bb_star))
        # print(np.sum(func**2))
        # TODO add constraints
        constraint = []

        return np.append(func, constraint)

    def forward_model(self, pars):
        '''
        Forward model: Gordon et al. (1988) reflectance plus chlorophyll
        fluorescence.

        :param pars: (chl, a_bg_ref, bb_bg_ref, S_bg, eta_bg)
        '''
        chl, a_bg_ref, bb_bg_ref, S_bg, eta_bg = pars

        # print(chl, a_bg_ref, bb_bg_ref, S_bg, eta_bg)

        a_bg = self.abg(a_bg_ref, S_bg)
        bb_bg = self.bbbg(bb_bg_ref, eta_bg)

        bb = self.bbw + self.bb_star * chl + bb_bg
        a = self.aw + self.a_star * chl + a_bg

        Rrs_fluo = water.Rrs2rrs(water.fluo_gower2004(chl, self.wl, Ed_ref=self.Ed_ref))
        Rrs = water.gordon88(a, bb)

        return Rrs + Rrs_fluo

    def call_solver(self, rrs, numpy=True):
        '''
        Run the least-squares inversion for one spectrum.

        :param rrs: spectrum (array, or pandas Series if ``numpy`` is False)
        :param numpy: set False if ``rrs`` is a pandas object
        :return: lmfit MinimizerResult
        '''
        if not numpy:
            rrs = rrs.values
        pars = lm.Parameters()
        pars.add('chl', value=0.5, min=0.03, max=600)
        pars.add('a_bg_ref', value=0.1, min=0, max=20)
        pars.add('bb_bg_ref', value=0.001, min=0, max=10)
        pars.add('S_bg', value=0.015, min=0.005, max=0.035)
        pars.add('eta_bg', value=0.5, min=0.2, max=1.2)

        min1 = lm.Minimizer(self.objfunc, pars, fcn_args=(rrs, self.a_star, self.bb_star))

        out1 = min1.least_squares(max_nfev=30, xtol=1e-7, ftol=1e-4)
        out1.params.pretty_print()

        # print(lm.fit_report(out1))
        return out1

    def multiprocess(self, df, nb_workers=10):
        '''
        Apply the solver to each row (spectrum) of a pandas DataFrame in
        parallel (requires the optional package ``pandarallel``).

        :param df: DataFrame of rrs spectra (one spectrum per row)
        :param nb_workers: number of processes
        :return: Series of lmfit results
        '''
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=nb_workers)
        res = df.parallel_apply(self.call_solver, axis=1, numpy=False)
        return res
