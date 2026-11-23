'''
List of input parameters to be put as metadata in the output netcdf file.

The classes of this module are simple containers (all attributes initialized
to ``None`` or empty lists) describing the atmosphere, air-water interface,
water body and bottom used to generate a radiative transfer look-up table.
'''


class scatterer:
    '''
    Optical properties of a population of scattering particles
    (hydrosols or aerosols).

    :ivar concentration: particle concentration
    :ivar model_type: name of the particle model
    :ivar mr: real part of the refractive index (relative to the medium)
    :ivar mi: imaginary part of the refractive index (relative to the medium)
    :ivar Cext_ref: extinction cross section at the reference wavelength
    :ivar Cext_wl: extinction cross section at the simulation wavelength
    :ivar Cscat_ref: scattering cross section at the reference wavelength
    :ivar Cscat_wl: scattering cross section at the simulation wavelength
    :ivar scat_angles: scattering angles (deg) of the scattering matrix
    :ivar P11: term 11 of the scattering matrix (phase function)
    :ivar P12: term 12 of the scattering matrix
    :ivar P22: term 22 of the scattering matrix
    :ivar P33: term 33 of the scattering matrix
    :ivar asymmetry_factor: asymmetry factor of the phase function
    '''
    def __init__(self):
        # - hydrosols
        self.concentration = None
        self.model_type = None
        self.mr = None
        self.mi = None

        self.Cext_ref = None
        self.Cext_wl = None
        self.Cscat_ref = None
        self.Cscat_wl = None

        self.scat_angles = []
        self.P11 = []
        self.P12 = []
        self.P22 = []
        self.P33 = []

        self.asymmetry_factor = None


# ------------------------
#    Atmosphere
# ------------------------
class atmosphere:
    '''
    Atmospheric parameters.

    :ivar rot: Rayleigh (molecular) optical thickness
    :ivar dpol: molecular depolarization ratio
    :ivar aerosol: aerosol parameters, with ``aot_ref`` (aerosol optical
        thickness at the reference wavelength) and ``aot_wl`` (aerosol
        optical thickness at the simulation wavelength)
    '''
    def __init__(self):
        # molecules:
        self.rot = None
        self.dpol = None

        class aerosol(scatterer):
            def __init__(self):
                self.aot_ref = None
                self.aot_wl = None

        self.aerosol = aerosol()


# ------------------------
#    Interface
# ------------------------
class interface:
    '''
    Air-water interface parameters.

    :ivar wind_speed: wind speed (m s-1)
    :ivar refractive_index: refractive index of water relative to air
    '''
    def __init__(self):
        self.wind_speed = None
        self.refractive_index = None


# ------------------------
#     Water
# ------------------------
class dissolved:
    '''
    Dissolved absorbing matter.

    :ivar a_cdom: CDOM absorption coefficient (m-1) at reference wavelength
    :ivar S_cdom: spectral slope of CDOM absorption (nm-1)
    :ivar a_cdim: absorption coefficient (m-1) of colored detrital
        inorganic matter at reference wavelength
    :ivar S_cdim: spectral slope of ``a_cdim`` (nm-1)
    '''
    def __init__(self):
        self.a_cdom = None
        self.S_cdom = None
        self.a_cdim = None
        self.S_cdim = None


class water:
    '''
    Water body constituents.

    :ivar dissolved: dissolved matter, see :class:`dissolved`
    :ivar phyto: phytoplankton particles, see :class:`scatterer`
    :ivar sediment: mineral particles, see :class:`scatterer`
    '''
    def __init__(self):

        self.dissolved = dissolved()
        self.phyto = scatterer()
        self.sediment = scatterer()


# ------------------------
#     Bottom
# ------------------------
class bottom:
    '''
    Sea bottom parameters.

    :ivar depth: bottom depth (m)
    :ivar type: bottom type (e.g., lambertian)
    :ivar albedo: bottom albedo
    '''
    def __init__(self):
        self.depth = None
        self.type = None
        self.albedo = None


class metadata:
    '''
    Full set of parameters of a radiative transfer simulation.

    :ivar wl_ref: reference wavelength
    :ivar wl: simulation wavelength
    :ivar truncature: truncation of the phase functions (``'yes'`` / ``'no'``)
    :ivar ig_max: maximum order of successive orders of scattering
    :ivar pressure: atmospheric pressure at sea level (hPa)
    :ivar aero_height: aerosol scale height
    :ivar ray_height: molecular (Rayleigh) scale height
    :ivar atmosphere: see :class:`atmosphere`
    :ivar water: see :class:`water`
    :ivar bottom: see :class:`bottom`
    '''

    def __init__(self):

        self.wl_ref = None
        self.wl = None

        self.truncature = ''  # (yes / no)
        self.ig_max = None

        self.pressure = None
        self.aero_height = None
        self.ray_height = None

        self.atmosphere = atmosphere()
        self.water = water()
        self.bottom = bottom()

