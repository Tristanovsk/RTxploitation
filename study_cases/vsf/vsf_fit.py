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



def P_RM(theta,g,alpha=0.5):
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

def P_TTRM(theta,gamma,g1,g2,alpha1=0.5,alpha2=0.5):
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
    forward = P_RM(theta,g1,alpha1)
    backward = P_RM(theta,g2,alpha2)
    return gamma * forward + (1-gamma) * backward

def objfunc(x, theta, vsf):
    '''
    Objective function to be minimized
    :param x: vector of unknowns
    :param theta: scattering angle
    :param vsf: phase function
    '''
    gamma, g1, g2, alpha1,alpha2 = np.array(list(x.valuesdict().values()))
    simu = P_TTRM(theta,gamma, g1, g2,alpha1,alpha2)
    return np.log(vsf) - np.log(simu)

for file in files:
    df = pd.read_csv(file,skiprows=8,sep='\t',index_col=0,skipinitialspace=True,na_values='inf')
    basename = os.path.basename(file).replace('.txt','')
    color=['blue','green','red','black']
    plt.figure()
    for i,(label,group) in enumerate(df.iteritems()):
        print(label)
        group_ = group.dropna()

        theta,vsf = group_.index.values,group_.values


        pars = lm.Parameters()
        pars.add('gamma', value=0.7, min=0, max=1)
        pars.add('g1', value=0.9, min=0, max=1)
        pars.add('g2', value=0.3, min=-1, max=0)
        pars.add('alpha1', value=0.5, min=-0.5, max=10)
        pars.add('alpha2', value=0.5, min=-0.5, max=10)
        min1 = lm.Minimizer(objfunc, pars, fcn_args=(theta,vsf))

        out1 = min1.least_squares() #max_nfev=30, xtol=1e-7, ftol=1e-4)
        out1.params.pretty_print()

        x = out1.x
        plt.plot(theta,vsf,color=color[i],label=label)
        plt.plot(theta,P_TTRM(theta,*x),'--',color=color[i])
    plt.legend()
    plt.xlabel('Scattering angle (deg)')
    plt.ylabel('Scattering function (-)')
    plt.suptitle(basename)
    plt.semilogy()
    plt.savefig(opj(dirfig,basename+'.png'),dpi=300)