"""
Simple parameterizations of atmospheric irradiance and water-leaving
signal (bio-optical relationships and chlorophyll fluorescence).
"""
import numpy as np
import Py6S


class atmo:
    '''
    Atmospheric parameterizations based on the 6S radiative transfer code
    (through the Py6S package), with a maritime aerosol model, sun at
    azimuth 0 and nadir viewing geometry.
    '''

    def __init__(self):
        pass

    def Ed_singlewl(self, sza, wl):
        '''
        Compute the downwelling irradiance at surface level for a single
        wavelength as the sum of the direct, diffuse and environmental
        irradiances simulated by 6S.

        :param sza: solar zenith angle in deg
        :param wl: wavelength in nm (size must be 1)
        :return: irradiance at surface level in W m-2 µm-1
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
        Compute the downwelling irradiance at surface level for several
        wavelengths as the sum of the direct, diffuse and environmental
        irradiances simulated by 6S.

        :param sza: solar zenith angle in deg
        :param wl: list or array of wavelengths in nm (size >= 2)
        :return: numpy array of irradiance at surface level in W m-2 µm-1
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
    '''
    Semi-analytical relationships between water inherent optical
    properties (IOPs) and remote sensing reflectance, and chlorophyll
    fluorescence models.

    :ivar g0: first coefficient of the Gordon et al. (1988) model
    :ivar g1: second coefficient of the Gordon et al. (1988) model
    '''
    def __init__(self):
        self.g0 = 0.089
        self.g1 = 0.125
        pass

    def gordon88(self, a, bb):
        '''
        Remote sensing reflectance from the Gordon et al. (1988) model:
        ``rrs = (g0 + g1 * u) * u`` with ``u = bb / (a + bb)``, converted into
        above-water reflectance with :meth:`rrs2Rrs`.

        :param a: total absorption coefficient (m-1)
        :param bb: total backscattering coefficient (m-1)
        :return: above-water remote sensing reflectance Rrs (sr-1)
        '''

        u = bb / (a + bb)
        rrs = (self.g0 + self.g1 * u) * u
        return self.rrs2Rrs(rrs)

    def inv_gordon88(self,rrs):
        '''
        Inversion of the Gordon et al. (1988) model: solve
        ``g1 * u**2 + g0 * u - rrs = 0`` for ``u = bb / (a + bb)``.
        Only the positive root is returned.

        :param rrs: below-water remote sensing reflectance (sr-1),
            see :meth:`Rrs2rrs`
        :return: ``u = bb / (a + bb)``
        '''
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
        '''
        Convert above-water into below-water remote sensing reflectance
        (Lee et al., 2002).

        :param Rrs: above-water remote sensing reflectance (sr-1)
        :return: below-water remote sensing reflectance rrs (sr-1)
        '''
        return Rrs / (0.52 + 1.7 * Rrs)

    def rrs2Rrs(self,rrs):
        '''
        Convert below-water into above-water remote sensing reflectance
        (Lee et al., 2002).

        :param rrs: below-water remote sensing reflectance (sr-1)
        :return: above-water remote sensing reflectance Rrs (sr-1)
        '''
        return 0.52 * rrs / (1 - 1.7 * rrs)

    def fluo_gower2004(self, chl, wl, Ed_ref=775., wl_ref=865):
        '''
        Simple Lfluo model from Gower et al. 2004
        (eq. 7.23 in Gilerson & Huot 2017 in
        Bio-optical modeling and remote sensing of inland waters (Elsevier)).

        The fluorescence radiance ``Lfluo`` (W m-2 µm-1 sr-1) is modeled as a
        Gaussian band centered on ``wl_ref`` with an amplitude depending
        on ``chl``, then divided by ``Ed_ref``.

        :param chl: chlorophyll concentration in mg m-3
        :param wl: wavelength in nm
        :param Ed_ref: downward irradiance (W m-2 µm-1) for fluo excitation
            at reference wavelength,
            can be obtained with ``Ed_ref = atmo().Ed_singlewl(sza, wl_ref)``
        :param wl_ref: reference wavelength (nm), center of the Gaussian
            emission band
        :return: fluorescence remote sensing reflectance Rrs_f (sr-1)
        '''


        L_ref = 0.15 * chl / (1 + 0.2 * chl)
        Lfluo = L_ref * np.exp(-4 * ((wl - wl_ref) / 25) ** 2)

        Rrs_f = Lfluo / Ed_ref

        return (Rrs_f)

        ## other

    def fluo_C(self, chl, wl, sza, wl_ref=685.):
        '''
        Lfluo model (eq. 7.13, 7.14, 7.27 in Gilerson & Huot 2017)
        accounting for the absorption by CDOM and non-algal particles,
        here derived from the phytoplankton absorption at 443 nm.
        The downward irradiance at ``wl_ref`` is computed with
        :meth:`atmo.Ed_singlewl`.

        :param chl: chlorophyll concentration in mg m-3
        :param wl: wavelength in nm
        :param sza: solar zenith angle in deg
        :param wl_ref: reference wavelength for fluo excitation (nm)
        :return: fluorescence remote sensing reflectance Rrs_f (sr-1)
        '''

        Ed_ref = atmo().Ed_singlewl(sza, wl_ref)
        a_phy443 = 0.042 * chl ** 0.8

        # TODO add a_cdom_nap as input value with appropriate spectral slope
        a_cdom_nap443 = 0.7 * a_phy443 + 0.56 * a_phy443

        L_ref= 0.092 * chl / (1 + 0.040 * a_cdom_nap443 + 0.078 * chl)
        Lfluo = L_ref * np.exp(-4 * ((wl - wl_ref) / 25) ** 2)
        Rrs_f = Lfluo / Ed_ref

        return (Rrs_f)








