''' modules dedicated to structure and exploit ancillary data (normally stored in aux folder)'''

import pandas as pd
from scipy.interpolate import interp1d


'''where you can set absolute and relative path used in the package'''
import os
import numpy as np

opj = os.path.join

root = os.path.dirname(os.path.abspath(__file__))
subdir = '../../data'
M2015_file = opj(root, subdir, 'rhoTable_Mobley2015.csv')
M1999_file = opj(root, subdir, 'rhoTable_Mobley1999.csv')
rhosoaa_fine_file = opj(root, subdir, 'surface_reflectance_factor_rho_fine_aerosol_rg0.06_sig0.46.csv')
rhosoaa_coarse_file = opj(root, subdir, 'surface_reflectance_factor_rho_coarse_aerosol_rg0.60_sig0.60.csv')
F0_file = opj(root, subdir, 'Thuillier_2003_0.3nm.dat')
water_scat_file = opj(root, subdir, 'water_coef.txt')
water_abs_file = opj(root, subdir,'purewater_abs_coefficients_v3.dat')


class iopw:
    def __init__(self):
        self.water_abs_file = water_abs_file
        self.water_scat_file = water_scat_file
        self.load_iopw()

    def load_iopw(self, ):
        self.iop_w = pd.read_csv(self.water_scat_file, skiprows=30, sep=' ', header=None,
                                 names=('wl', 'a', 'b')).set_index('wl').to_xarray()
        self.water_data = pd.read_csv(self.water_abs_file, skiprows=4, sep='\t').set_index('wl').to_xarray()

    def get_iopw(self, wl, Twater=20,mute=False):
        '''
        interpolate and return absorption and back-scattering coefficients (m-1)
        for pure water
        :param wl: wavelength in nm, scalar or np.array
        :param Twater: water temperature in deg C
        :param mute: if true values are not returned (only saved in object)
        :return:
        '''
        Twater_ref=20
        self.wl = wl
        a_coef_T = self.water_data.a_coef + (Twater - Twater_ref)*self.water_data.PsiT
        self.aw = a_coef_T.interp(wl=wl)
        self.bbw = self.iop_w.b.interp(wl=wl,kwargs={"fill_value": 'extrapolate'})/ 2.
        if not mute:
            return self.aw, self.bbw


class irradiance:
    def __init__(self, F0_file=F0_file):
        self.F0_file = F0_file

    def load_F0(self, ):
        self.F0df = pd.read_csv(self.F0_file, skiprows=15, sep='\t', header=None, names=('wl', 'F0'))

    def get_F0(self, wl, mute=False):
        '''
        interpolate and return solar spectral irradiance (mW/m2/nm)

        :param wl: wavelength in nm, scalar or np.array
        :param mute: if true values are not returned (only saved in object)
        :return:
        '''
        self.wl = wl
        self.F0 = interp1d(self.F0df.wl, self.F0df.F0, fill_value='extrapolate')(wl)

        if not mute:
            return self.F0



class cdom:
    def __init__(self, a440=0., wl=[440]):
        self.wl = wl
        self.a440 = a440
        self.S440 = 0.018

    def get_acdom(self):
        self.a = self.a440 * np.exp(-self.S440 * (self.wl - 440.))
        return self.a
