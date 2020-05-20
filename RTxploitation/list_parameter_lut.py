''' list of input parameters to be put as metadata in the output netcdf file'''


class scatterer:
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
    def __init__(self):
        self.wind_speed = None
        self.refractive_index = None


# ------------------------
#     Water
# ------------------------
class dissolved:
    def __init__(self):
        self.a_cdom = None
        self.S_cdom = None
        self.a_cdim = None
        self.S_cdim = None


class water:
    def __init__(self):

        self.dissolved = dissolved()
        self.phyto = scatterer()
        self.sediment = scatterer()


# ------------------------
#     Bottom
# ------------------------
class bottom:
    def __init__(self):
        self.depth = None
        self.type = None
        self.albedo = None


class metadata:

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


