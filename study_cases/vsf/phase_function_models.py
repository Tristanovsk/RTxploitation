import os
opj =os.path.join
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import lmfit as lm

import RTxploitation as rtx

dir = opj(rtx.__path__[0],'..','study_cases','vsf',)
dirdata = opj(dir, 'data')
dirfig = opj(dir,'fig')

files =glob.glob(opj(dirdata,'normalized_vsf*txt'))


class models:
    def __init__(self):
        pass

    #-------------------------------------------
    # single term models
    #-------------------------------------------

    def P_RM(self,theta,g,alpha=0.5):
        '''
        Phase function from Reynold and McCormick (J. Opt. Soc. Am., 1980,
        70, 1206–1212), phase function approximation which described
        highly anisotropic angular scattering distributions
        :param theta: scattering angle (deg)
        :param g: asymmetry factor
        :param alpha: fitting parameter
        :return:
        '''
        theta_rad = np.radians(theta)
        num = alpha * g * (1. - g**2)**(2*alpha)
        denom = np.pi * (1+g**2-2*g*np.cos(theta_rad))**(alpha+1) * \
                ((1+g)**(2*alpha)-(1-g)**(2*alpha))
        return num / denom

    def P_HG(self,theta,g):
        '''
        When \alpha equals 0.5, P_RM reduces to the Henyey–Greenstein (HG)
        phase function (L. C. Henyey and J. L. Greenstein,
         Astrophys. J., 1941, 93, 70–83.)
        :param theta: scattering angle (deg)
        :param g: asymmetry factor
        :return:
        '''

        return self.P_RM(theta,g,alpha=0.5)

    def P_FF(self,theta,n,m):
        '''
        Forand–Fournier phase function (G. R. Fournier and J. L. Forand,
        Proc. SPIE, 1994, 2258, 194–201.)
        :param theta: scattering angle (deg)
        :param n: relative refractive index (real part), 1 < n < 1.35
        :param m: slope of power-law size distribution, 3.5 < m < 5.
        :return:
        '''

        theta_rad = np.radians(theta)
        u = 2 * np.sin(theta_rad/2)
        nu = (3-m)/2
        delta = u**2/(3*(n-1)**2)
        delta_pi = 4/(3*(n-1)**2)
        delta_nu = delta**nu
        delta_pi_nu=delta_pi**nu
        pff = 1/ (4*np.pi*(1-delta)**2*delta_nu) * \
              ( (nu*(1-delta)-(1-delta_nu)) + 4/u**2*(delta*(1-delta_nu)-nu*(1-delta))) \
              - 1/(16*np.pi)*(1-delta_pi_nu)*(3*np.cos(theta_rad)**2-1)/((1-delta_pi)*delta_pi_nu)
        return pff

    #-------------------------------------------
    # two-term models
    #-------------------------------------------

    def P_TTRM(self,theta,gamma,g1,g2,alpha1=0.5,alpha2=0.5):
        '''
        This function has two parts with two different asymmetry
        factors, where g 1 is positive and g 2 is negative, in order to treat
        the forward and backward peaks in the phase function. The
        parameter γ gives the forward scattering portion while (1 − γ)
        is the backward scattering portion.
        :param theta: scattering angle (deg)
        :param gamma: forward/backward ratio
        :param g1: asymmetry factor >0
        :param g2: asymmetry factor <0
        :param alpha: fitting parameter
        :param alpha2: fitting parameter
        :return:
        '''
        forward = self.P_RM(theta,g1,alpha1)
        backward = self.P_RM(theta,g2,alpha2)
        return gamma * forward + (1-gamma) * backward

    def P_TTFF(self,theta,gamma,n1,m1,n2,m2):
        '''
        Two-term Fournier-Forand phase function with two particle groups
        defined by their refractive index (real part) and power law slope of size distribution
        :param theta: scattering angle (deg)
        :param gamma: ratio of the first particle group
        :param n1: refractive index of group 1
        :param m1: size distribution slope of group 1
        :param n2: refractive index of group 2
        :param m2: size distribution slope of group 2
        :return:
        '''
        group1 = self.P_FF(theta,n1,m1)
        group2 = self.P_FF(theta,n2,m2)
        return gamma * group1 + (1-gamma) * group2