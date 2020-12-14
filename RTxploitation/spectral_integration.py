import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import Py6S as sixs


class spectral_responses:

    # class rsr:
    #     def __init__(self):
    #         self.wl = []
    #         self.rsr = []

    def __init__(self, sat=None,band_idx=None):
        '''

        :param sat: satellite sensor ID: 'oli', 's2a',..., 's3b, 'modisa'...
        :param band_idx: if None all bands considered, list of band index otherwise
        '''
        # get info from Py6S package
        self.lut = sixs.PredefinedWavelengths
        # wavelength increment used to generate sensor spectral responses through 6SV and Py6S
        self.inc = 0.0025
        # debug np.arange for float rounding
        self.debug_arange = 5.1e-4

        self.satellites = []
        self.get = {'oli': self.get_landsat_oli(),
                    's2a': self.get_s2a_msi(),
                    's2b': self.get_s2b_msi(),
                    's3a': self.get_s3a_olci(),
                    's3b': self.get_s3b_olci(),
                    # 'meris': self.get_meris(),
                    'modisa': self.get_modisa(),
                    'modist': self.get_modist()}
        self.satellites = list(self.get.keys())

        if sat != None:
            self.satellite = sat
            self.get_rsr(sat, band_idx)

    def get_rsr(self, sat, band_idx=None):
        '''
        Get LUT relative spectral response of the bands of `sat`.
        Note that wavelengths are converted into nm.
        :param sat: satellite sensor ID: 'oli', 's2a',..., 's3b, 'modisa'...
        :param band_idx: if None all bands considered, list of band index otherwise
        :return:
        '''

        info = self.get[sat]
        if band_idx != None:
            info =info[band_idx]

        rsrs = []
        for void, start, end, rsr_ in info:
            wl = np.arange(start, end + self.debug_arange, self.inc) * 1e3
            rsr = xr.DataArray(np.array(rsr_), coords=[('wl', wl)])
            rsrs.append(rsr)
        self.rsrs = np.array(rsrs)
        return self.rsrs

    def convolution(self, wl, param):
        '''
        Integrate/convolve param values for each spectral band based on the `sat` lut values.
        :param wl: wavelengths of param, np.array in nm
        :param param: param, np.array
        :return:
        '''

        param_rsr = []
        for rsr in self.rsrs:
            idx = (wl >= rsr.wl[0].values) & (wl <= rsr.wl[-1].values)
            wl_, param_ = wl[idx], param[idx]
            rsr_ = rsr.interp(wl=wl_)

            param_rsr.append(
                np.trapz(param_ * rsr_.data, rsr_.wl) /
                rsr_.integrate('wl'))

        return np.array(param_rsr)

    def xr_convolution(self, xr_param):
        wl,param = xr_param.wl.values,xr_param.data
        return self.convolution(wl,param)

    def plot_all_sat(self):
        sats = self.satellites
        fig, axs = plt.subplots(nrows=len(sats), figsize=[15, 25])
        axs = axs.ravel()
        for ax, sat in zip(axs, sats):
            print(sat)
            rsrs = self.get_rsr(sat)
            for rsr in rsrs:
                ax.plot(rsr.wl, rsr.data)
            ax.set_title(sat)
        plt.tight_layout()
        plt.show()
        return fig, axs

    def get_landsat_oli(self):
        return np.array([self.lut.LANDSAT_OLI_B1,
                self.lut.LANDSAT_OLI_B2,
                self.lut.LANDSAT_OLI_B3,
                self.lut.LANDSAT_OLI_B4,
                self.lut.LANDSAT_OLI_B5,
                self.lut.LANDSAT_OLI_B6,
                self.lut.LANDSAT_OLI_B7,
                self.lut.LANDSAT_OLI_B8,
                self.lut.LANDSAT_OLI_B9])

    def get_s2a_msi(self):
        return np.array([self.lut.S2A_MSI_01,
                self.lut.S2A_MSI_02,
                self.lut.S2A_MSI_03,
                self.lut.S2A_MSI_04,
                self.lut.S2A_MSI_05,
                self.lut.S2A_MSI_06,
                self.lut.S2A_MSI_07,
                self.lut.S2A_MSI_08,
                self.lut.S2A_MSI_8A,
                self.lut.S2A_MSI_09,
                self.lut.S2A_MSI_10,
                self.lut.S2A_MSI_11,
                self.lut.S2A_MSI_12])

    def get_s2b_msi(self):
        return np.array([self.lut.S2B_MSI_01,
                self.lut.S2B_MSI_02,
                self.lut.S2B_MSI_03,
                self.lut.S2B_MSI_04,
                self.lut.S2B_MSI_05,
                self.lut.S2B_MSI_06,
                self.lut.S2B_MSI_07,
                self.lut.S2B_MSI_08,
                self.lut.S2B_MSI_8A,
                self.lut.S2B_MSI_09,
                self.lut.S2B_MSI_10,
                self.lut.S2B_MSI_11,
                self.lut.S2B_MSI_12])

    def get_s3a_olci(self):
        return np.array([self.lut.S3A_OLCI_01,
                self.lut.S3A_OLCI_02,
                self.lut.S3A_OLCI_03,
                self.lut.S3A_OLCI_04,
                self.lut.S3A_OLCI_05,
                self.lut.S3A_OLCI_06,
                self.lut.S3A_OLCI_07,
                self.lut.S3A_OLCI_08,
                self.lut.S3A_OLCI_09,
                self.lut.S3A_OLCI_10,
                self.lut.S3A_OLCI_11,
                self.lut.S3A_OLCI_12,
                self.lut.S3A_OLCI_13,
                self.lut.S3A_OLCI_14,
                self.lut.S3A_OLCI_15,
                self.lut.S3A_OLCI_16,
                self.lut.S3A_OLCI_17,
                self.lut.S3A_OLCI_18,
                self.lut.S3A_OLCI_19,
                self.lut.S3A_OLCI_20,
                self.lut.S3A_OLCI_21])

    def get_s3b_olci(self):
        return np.array([self.lut.S3B_OLCI_01,
                self.lut.S3B_OLCI_02,
                self.lut.S3B_OLCI_03,
                self.lut.S3B_OLCI_04,
                self.lut.S3B_OLCI_05,
                self.lut.S3B_OLCI_06,
                self.lut.S3B_OLCI_07,
                self.lut.S3B_OLCI_08,
                self.lut.S3B_OLCI_09,
                self.lut.S3B_OLCI_10,
                self.lut.S3B_OLCI_11,
                self.lut.S3B_OLCI_12,
                self.lut.S3B_OLCI_13,
                self.lut.S3B_OLCI_14,
                self.lut.S3B_OLCI_15,
                self.lut.S3B_OLCI_16,
                self.lut.S3B_OLCI_17,
                self.lut.S3B_OLCI_18,
                self.lut.S3B_OLCI_19,
                self.lut.S3B_OLCI_20,
                self.lut.S3B_OLCI_21])

    def get_meris(self):
        return np.array([self.lut.MERIS_B1, self.lut.MERIS_B2, self.lut.MERIS_B3,
                self.lut.MERIS_B4, self.lut.MERIS_B5, self.lut.MERIS_B6,
                self.lut.MERIS_B7, self.lut.MERIS_B9, self.lut.MERIS_B10,
                self.lut.MERIS_B11, self.lut.MERIS_B12, self.lut.MERIS_B8,
                self.lut.MERIS_B13, self.lut.MERIS_B14, self.lut.MERIS_B15])

    def get_modisa(self):
        return np.array([self.lut.ACCURATE_MODIS_AQUA_1,
                self.lut.ACCURATE_MODIS_AQUA_2,
                self.lut.ACCURATE_MODIS_AQUA_3,
                self.lut.ACCURATE_MODIS_AQUA_4,
                self.lut.ACCURATE_MODIS_AQUA_5,
                self.lut.ACCURATE_MODIS_AQUA_6,
                self.lut.ACCURATE_MODIS_AQUA_7,
                self.lut.ACCURATE_MODIS_AQUA_11,
                self.lut.ACCURATE_MODIS_AQUA_12,
                self.lut.ACCURATE_MODIS_AQUA_13,
                self.lut.ACCURATE_MODIS_AQUA_14,
                self.lut.ACCURATE_MODIS_AQUA_15])

    def get_modist(self):
        return np.array([self.lut.ACCURATE_MODIS_TERRA_1,
                self.lut.ACCURATE_MODIS_TERRA_2,
                self.lut.ACCURATE_MODIS_TERRA_3,
                self.lut.ACCURATE_MODIS_TERRA_4,
                self.lut.ACCURATE_MODIS_TERRA_5,
                self.lut.ACCURATE_MODIS_TERRA_6,
                self.lut.ACCURATE_MODIS_TERRA_7,
                self.lut.ACCURATE_MODIS_TERRA_11,
                self.lut.ACCURATE_MODIS_TERRA_12,
                self.lut.ACCURATE_MODIS_TERRA_13,
                self.lut.ACCURATE_MODIS_TERRA_14,
                self.lut.ACCURATE_MODIS_TERRA_15])
