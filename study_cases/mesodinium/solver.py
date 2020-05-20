import numpy as np
import lmfit as lm
import pandas as pd
from pandarallel import pandarallel



from RTxploitation import parameterization as RTp
import RTxploitation.auxdata as ad

water = RTp.water()


class Rrs_inversion:
    def __init__(self, wl, a_star, bb_star, sza):
        '''

        :param wl: wavelength of spectral values (i.e., arrays) in nm
        '''
        self.wl = wl
        self.a_star = a_star
        self.bb_star = bb_star
        self.aw, self.bbw = ad.iopw().get_iopw(wl)

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
        chl, a_bg_ref, bb_bg_ref, S_bg, eta_bg = pars

        # print(chl, a_bg_ref, bb_bg_ref, S_bg, eta_bg)

        a_bg = self.abg(a_bg_ref, S_bg)
        bb_bg = self.bbbg(bb_bg_ref, eta_bg)

        bb = self.bbw + self.bb_star * chl + bb_bg
        a = self.aw + self.a_star * chl + a_bg

        Rrs_fluo = water.Rrs2rrs(water.fluo_gower2004(chl, self.wl,  Ed_ref=self.Ed_ref))
        Rrs = water.gordon88(a, bb)

        return Rrs + Rrs_fluo

    def call_solver(self, rrs):

        rrs = rrs.values
        pars = lm.Parameters()
        pars.add('chl', value=0.05, min=0, max=600)
        pars.add('a_bg_ref', value=0.1, min=0, max=20)
        pars.add('bb_bg_ref', value=0.001, min=0, max=10)
        pars.add('S_bg', value=0.015, min=0.005, max=0.035)
        pars.add('eta_bg', value=0.5, min=0.2, max=1.2)

        min1 = lm.Minimizer(self.objfunc, pars, fcn_args=(rrs, self.a_star, self.bb_star))

        out1 = min1.least_squares(max_nfev=30, xtol=1e-7, ftol=1e-4)
        out1.params.pretty_print()

        print(lm.fit_report(out1))
        return out1

    def multiprocess(self, df):
        pandarallel.initialize(nb_workers=10)
        res = df.parallel_apply(self.call_solver, axis=1)
        return res
