import numpy as np
import Py6S


class atmo:

    def __init__(self):
        pass

    def Ed_singlewl(self, sza, wl):
        '''

        :param sza: solar zenith angle in deg
        :param wl: wavelength in nm (size must be 1)
        :return: irradiance at surface level in w m-2 mic-1
        '''


        s = Py6S.SixS()
        s.geometry.solar_z = sza
        s.geometry.solar_a = 0
        s.geometry.view_z = 0
        s.geometry.view_a = 0
        s.aero_profile = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.Maritime)
        s.wavelength = Py6S.Wavelength(wl / 1000)
        s.run()

        irradiance = s.outputs.diffuse_solar_irradiance + s.outputs.direct_solar_irradiance \
                     + s.outputs.environmental_irradiance

        return irradiance

    def Ed_multiwl(self, sza, wl):
        '''

        :param sza: solar zenith angle in deg
        :param wl: list of wavelengths in nm (size >= 2)
        :return: irradiance at surface level in w m-2 mic-1
        '''

        if not isinstance(wl, np.ndarray):
            wl = np.array(wl)
        s = Py6S.SixS()
        s.geometry.solar_z = sza
        s.geometry.solar_a = 0
        s.geometry.view_z = 0
        s.geometry.view_a = 0
        s.aero_profile = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.Maritime)
        wavelengths, results = Py6S.SixSHelpers.Wavelengths.run_wavelengths(s, wl / 1000)

        F0, trans_gas, irradiance = [], [], []
        for i in range(len(wl)):
            res = results[i]
            F0 = np.append(F0, res.solar_spectrum)
            trans_gas = np.append(trans_gas, res.total_gaseous_transmittance)
            irradiance = np.append(irradiance,
                                   res.diffuse_solar_irradiance + res.direct_solar_irradiance +
                                   res.environmental_irradiance)
        return irradiance


class water:
    def __init__(self):
        self.g0 = 0.089
        self.g1 = 0.125
        pass

    def gordon88(self, a, bb):

        u = bb / (a + bb)
        rrs = (self.g0 + self.g1 * u) * u
        return self.rrs2Rrs(rrs)

    def inv_gordon88(self,rrs):
        deltas = self.g0**2 + 4 * self.g1 * rrs
        #print('delta',deltas)
        # roots = []
        # for delta in deltas:
        #     if delta >=0.:
        #         roots.append( [(-self.g0 + np.sqrt(delta)) / (2*self.g1),
        #                  (-self.g0 - np.sqrt(delta)) / (2*self.g1)] )
        #     else:
        #         roots.append( [np.nan,np.nan])
        # simplification first root is the positive one
        roots = (-self.g0 + np.sqrt(deltas)) / (2*self.g1)
        return roots

    def Rrs2rrs(self,Rrs):
        return Rrs / (0.52 + 1.7 * Rrs)

    def rrs2Rrs(self,rrs):
        return 0.52 * rrs / (1 - 1.7 * rrs)

    def fluo_gower2004(self, chl, wl, Ed_ref=775., wl_ref=865):
        '''
        simple Lfluo model from Gower et al. 2004
        (eq. 7.23 in Gilerson & Huot 2017 in
         Bio-optical modeling and remote sensing of inland waters (Elsevier))

        :param chl: chlorophyll concentration in mg m-3
        :param wl: wavelength in nm
        :param sza: solar zenith angle in deg
        :param Ed_ref: downward irradiance for fluo excitation
               at reference wavelength (nm),
               can be obtained with Ed_ref = atmo().Ed_singlewl(sza, wl_ref)
        Lfluo in W m-2 μm-1 sr-1
        Ed_ref in
        :return:
        '''


        L_ref = 0.15 * chl / (1 + 0.2 * chl)
        Lfluo = L_ref * np.exp(-4 * ((wl - wl_ref) / 25) ** 2)

        Rrs_f = Lfluo / Ed_ref

        return (Rrs_f)

        ## other

    def fluo_C(self, chl, wl, sza, wl_ref=685.):
        '''
        Lfluo model (eq. 7.13, 7.14, 7.27 in Gilerson & Huot 2017)
        :param chl: chlorophyll concentration in mg m-3
        :param wl: wavelength in nm
        :param sza: solar zenith angle in deg
        :param wl_ref: reference wavelength for fluo excitation (nm)
        :return:
        '''

        Ed_ref = atmo().Ed_singlewl(sza, wl_ref)
        a_phy443 = 0.042 * chl ** 0.8

        # TODO add a_cdom_nap as input value with appropriate spectral slope
        a_cdom_nap443 = 0.7 * a_phy443 + 0.56 * a_phy443

        L_ref= 0.092 * chl / (1 + 0.040 * a_cdom_nap443 + 0.078 * chl)
        Lfluo = L_ref * np.exp(-4 * ((wl - wl_ref) / 25) ** 2)
        Rrs_f = Lfluo / Ed_ref

        return (Rrs_f)








