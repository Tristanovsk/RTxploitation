'''
Modules dedicated to structure and exploit ancillary data (normally stored
in the ``data`` folder): pure water optical properties, extraterrestrial
solar irradiance (Thuillier et al., 2003), sea surface reflectance factor
tables and CDOM absorption.
'''

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
    '''
    Inherent optical properties of pure water.
    Data are loaded at initialization.
    '''
    def __init__(self):
        self.water_abs_file = water_abs_file
        self.water_scat_file = water_scat_file
        self.load_iopw()

    def load_iopw(self, ):
        '''
        Load pure water absorption (with temperature correction coefficient
        ``PsiT``) and scattering coefficients from data files.
        '''
        self.iop_w = pd.read_csv(self.water_scat_file, skiprows=30, sep=' ', header=None,
                                 names=('wl', 'a', 'b')).set_index('wl').to_xarray()
        self.water_data = pd.read_csv(self.water_abs_file, skiprows=4, sep='\t').set_index('wl').to_xarray()

    def get_iopw(self, wl, Twater=20,mute=False):
        '''
        Interpolate and return absorption and back-scattering coefficients (m-1)
        for pure water.

        Absorption is corrected for temperature as
        ``a(T) = a(20) + (T - 20) * PsiT``; backscattering is half the
        scattering coefficient.

        :param wl: wavelength in nm, scalar or np.array
        :param Twater: water temperature in deg C
        :param mute: if true values are not returned (only saved in object)
        :return: tuple (aw, bbw) of :class:`xarray.DataArray`
        '''
        Twater_ref=20
        self.wl = wl
        a_coef_T = self.water_data.a_coef + (Twater - Twater_ref)*self.water_data.PsiT
        self.aw = a_coef_T.interp(wl=wl)
        self.bbw = self.iop_w.b.interp(wl=wl,kwargs={"fill_value": 'extrapolate'})/ 2.
        if not mute:
            return self.aw, self.bbw


class irradiance:
    '''
    Extraterrestrial solar irradiance.

    :param F0_file: path of the solar spectrum file
        (default: Thuillier et al., 2003)
    '''
    def __init__(self, F0_file=F0_file):
        self.F0_file = F0_file

    def load_F0(self, ):
        '''
        Load the solar spectrum into ``self.F0df``; must be called before
        :meth:`get_F0`.
        '''
        self.F0df = pd.read_csv(self.F0_file, skiprows=15, sep='\t', header=None, names=('wl', 'F0'))

    def get_F0(self, wl, mute=False):
        '''
        Interpolate and return solar spectral irradiance (mW/m2/nm).

        :param wl: wavelength in nm, scalar or np.array
        :param mute: if true values are not returned (only saved in object)
        :return: solar irradiance at ``wl``
        '''
        self.wl = wl
        self.F0 = interp1d(self.F0df.wl, self.F0df.F0, fill_value='extrapolate')(wl)

        if not mute:
            return self.F0



class cdom:
    '''
    Absorption of colored dissolved organic matter (CDOM) with an
    exponential spectral shape
    ``a(wl) = a440 * exp(-S440 * (wl - 440))`` with ``S440 = 0.018`` nm-1.

    :param a440: CDOM absorption coefficient at 440 nm (m-1)
    :param wl: wavelengths in nm (np.array)
    '''
    def __init__(self, a440=0., wl=[440]):
        self.wl = wl
        self.a440 = a440
        self.S440 = 0.018

    def get_acdom(self):
        '''
        :return: CDOM absorption coefficient (m-1) at ``self.wl``
        '''
        self.a = self.a440 * np.exp(-self.S440 * (self.wl - 440.))
        return self.a
