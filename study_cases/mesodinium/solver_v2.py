import numpy as np
import xarray as xr
import lmfit as lm
import pandas as pd
from pandarallel import pandarallel

from Py6S import *

import RTxploitation as RT

RTp = RT.parameterization
ad = RT.auxdata
spectral = RT.spectral_integration.spectral_responses
water = RTp.water()


class Rrs_inversion:
    def __init__(self, a_star: xr.DataArray, bb_star: xr.DataArray, sza, sat='s2a', band_idx=None, wl_=None):
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
            rsrs = spectral(sat, band_idx)
            wl_ = np.arange(350, 2700, 1)

            self.wl = rsrs.convolution(wl_, wl_)
            self.a_star = rsrs.xr_convolution(a_star)
            self.bb_star = rsrs.xr_convolution(bb_star)
            aw, bbw = ad.iopw().get_iopw(wl_)
            self.aw, self.bbw = rsrs.convolution(wl_, aw), rsrs.convolution(wl_, bbw)

        # measurements uncertainty for each band (strictly positive)
        self.sig_meas = [1] * len(self.wl)

        # parameters for the Gordon88 model
        self.g0 = 0.089
        self.g1 = 0.125
        self.g0sq = self.g0 ** 2

        self.Ed_ref = RTp.atmo().Ed_singlewl(sza, 865)

    def Rrs_to_iops(self, Rrs, wl_nir_threshold=715):
        '''

        :param Rrs: Remote sensing reflectance above water
        :return:
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

        :param a_bg_ref:
        :param S_bg:
        :param wl_ref:
        :return:
        '''
        return a_bg_ref * np.exp(-S_bg * (self.wl - wl_ref))

    def bbbg(self, bb_bg_ref, eta_bg, wl_ref=440):
        '''

        :param bb_bg_ref:
        :param eta_bg:
        :param wl_ref:
        :return:
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

        pars = np.array(list(x.valuesdict().values()))

        rrs_simu = self.forward_model(pars)

        func = (rrs - rrs_simu)  / self.sig_meas

        # TODO add constraints
        constraint = []

        return np.append(func, constraint)

    def forward_model(self, pars, level='below'):
        chl, a_bg_ref, bb_bg_ref, S_bg, eta_bg = pars

        # print(chl, a_bg_ref, bb_bg_ref, S_bg, eta_bg)

        a_bg = self.abg(a_bg_ref, S_bg)
        bb_bg = self.bbbg(bb_bg_ref, eta_bg)

        bb = self.bbw + self.bb_star * chl + bb_bg
        a = self.aw + self.a_star * chl + a_bg

        rrs_fluo = water.Rrs2rrs(water.fluo_gower2004(chl, self.wl, Ed_ref=self.Ed_ref))
        rrs = water.gordon88(a, bb)
        rrs_tot = rrs + rrs_fluo
        if level == "below":
            return rrs_tot
        else:
            return water.rrs2Rrs(rrs_tot)

    def call_solver(self, rrs, numpy=True, xinit=[0.5,0.1,0.1,0.015,0.5]):
        if not numpy:
            rrs = rrs.values
        pars = lm.Parameters()
        pars.add('chl', value=xinit[0], min=0.03, max=600)
        pars.add('a_bg_ref', value=xinit[1], min=0, max=60)
        pars.add('bb_bg_ref', value=xinit[2], min=0, max=60)
        pars.add('S_bg', value=xinit[3], min=0.005, max=0.035)
        pars.add('eta_bg', value=xinit[4], min=0.2, max=1.2)

        min1 = lm.Minimizer(self.objfunc, pars, fcn_args=(rrs, self.a_star, self.bb_star))

        out1 = min1.least_squares()#max_nfev=50, xtol=1e-7, ftol=1e-7)
        out1.params.pretty_print()

        # print(lm.fit_report(out1))
        return out1

    def multiprocess(self, df):
        pandarallel.initialize(nb_workers=10)
        res = df.parallel_apply(self.call_solver, axis=1)
        return res
