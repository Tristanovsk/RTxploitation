''' modules dedicated to structure and exploit ancillary data (normally stored in aux folder)'''

import pandas as pd
from scipy.interpolate import interp1d


'''where you can set absolute and relative path used in the package'''
import os
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))

M2015_file = os.path.join(root, '../../data/rhoTable_Mobley2015.csv')
M1999_file = os.path.join(root, '../../data/rhoTable_Mobley1999.csv')
rhosoaa_fine_file = os.path.join(root, '../../data/surface_reflectance_factor_rho_fine_aerosol_rg0.06_sig0.46.csv')
rhosoaa_coarse_file = os.path.join(root, '../../data/surface_reflectance_factor_rho_coarse_aerosol_rg0.60_sig0.60.csv')
iopw_file =  os.path.join(root, '../../data/water_coef.txt')
F0_file =  os.path.join(root, '../../data/Thuillier_2003_0.3nm.dat')


class iopw:
    def __init__(self, iopw_file=iopw_file):
        self.iopw_file = iopw_file
        self.load_iopw()

    def load_iopw(self, ):
        self.iop_w = pd.read_csv(self.iopw_file, skiprows=30, sep=' ', header=None, names=('wl', 'a', 'bb'))

    def get_iopw(self, wl, mute=False):
        '''
        interpolate and return absorption and back-scattering coefficients (m-1)
        for pure water
        :param wl: wavelength in nm, scalar or np.array
        :param mute: if true values are not returned (only saved in object)
        :return:
        '''
        self.wl = wl
        self.aw = interp1d(self.iop_w.wl, self.iop_w.a, fill_value='extrapolate')(wl)
        self.bbw = interp1d(self.iop_w.wl, self.iop_w.bb / 2., fill_value='extrapolate')(wl)
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
