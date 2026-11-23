"""
Plotting helpers for the Mesodinium case study.
"""
import os
import numpy as np

# ----------------------------
# set plotting styles
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.ioff()
mpl.rcParams.update({'font.size': 18})

from RTxploitation import parameterization as RTp

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

class Rrs:
    '''Figures of forward-modeled Rrs and IOPs.'''
    def __init__(self):
        pass

    def plot_Rrs_a_bb(self, wl, a_star, bbp_star, aw, bbw, a_bg, bbp_bg, coef=1,
                      param_star=np.logspace(np.log10(0.1), np.log10(200), 50),
                      cmin=0, cmax=200, title='Chl --> Rrs (model direct)', figdir='./',
                      Ed_ref=775.):
        '''
        Plot Rrs (Gordon et al. 1988 + fluorescence), total absorption and
        backscattering for a range of chl-a concentrations, for pure algae
        (top) and algae + background (bottom).

        :param wl: wavelengths (nm)
        :param a_star: chl-specific absorption of the algae (m2 mg-1)
        :param bbp_star: chl-specific backscattering of the algae (m2 mg-1)
        :param aw: pure water absorption (m-1)
        :param bbw: pure water backscattering (m-1)
        :param a_bg: background absorption (m-1)
        :param bbp_bg: background backscattering (m-1)
        :param coef: scaling factor applied to ``bbp_star``
        :param param_star: chl-a concentrations (mg m-3)
        :param cmin: min of the color scale
        :param cmax: max of the color scale
        :param title: figure title
        :param figdir: output directory
        :param Ed_ref: downwelling irradiance at 865 nm for fluorescence (W m-2 um-1)
        '''
        water = RTp.water()

        cmap = mpl.colormaps['Spectral'].reversed()

        norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 24))  # ncols=3, figsize=(30, 12))
        fig.subplots_adjust(left=0.1, right=0.9, hspace=.5, wspace=0.45)
        axs = axs.T.ravel()

        for i in [0, 3]:

            for chl in param_star:

                Rrs_fluo = water.fluo_gower2004(chl, wl, Ed_ref=Ed_ref)
                if i == 0:
                    Rrs_g88 = water.gordon88(aw + a_star * chl,
                                             bbw + coef * bbp_star * chl)
                    subtitle = 'Pure M. Rubrum'
                    # plot iop
                    axs[i + 1].plot(wl, aw, 'k--', label='aw', lw=2.5, alpha=0.75)
                    axs[i + 1].plot(wl, a_star * chl + aw, color=cmap(norm(chl)), label='atot', lw=2.5,
                                    alpha=0.75)

                    axs[i + 2].plot(wl, bbw, 'k--', label='bbw', lw=2.5, alpha=0.75)
                    axs[i + 2].plot(wl, coef * bbp_star * chl + bbw, color=cmap(norm(chl)), label='bb',
                                    lw=2.5,
                                    alpha=0.75)
                else:
                    Rrs_g88 = water.gordon88(aw + a_star * chl +  a_bg,
                                             bbw + coef * bbp_star * chl + bbp_bg)
                    subtitle = 'M. Rubrum + background'
                    axs[i + 1].plot(wl, aw, 'k--', label='aw', lw=2.5, alpha=0.75)
                    axs[i + 1].plot(wl, a_star * chl + aw + a_bg,
                                    color=cmap(norm(chl)),
                                    label='atot', lw=2.5, alpha=0.75)
                    axs[i + 2].plot(wl, bbw, 'k--', label='bbw', lw=2.5, alpha=0.75)
                    axs[i + 2].plot(wl, coef * bbp_star * chl + bbw + bbp_bg, color=cmap(norm(chl)),
                                    label='bb', lw=2.5, alpha=0.75)

                # y = group.values
                # ax.plot(wl, y, color=cmap(norm(chl)), lw=2.5, alpha=0.75)
                axs[i].plot(wl, Rrs_g88 + Rrs_fluo, color=cmap(norm(chl)), lw=2.5, alpha=0.75)
                # ax.plot(wl,Rrs_fluo,'--')
                axs[i].set_title(subtitle)
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(sm, cax=cax, format=mpl.ticker.ScalarFormatter(),
                                shrink=1.0, fraction=0.1, pad=0)

            axs[i].set_ylabel(r'$R_{rs}\  (sr^{-1})$')
            axs[i].set_xlabel(r'Wavelength (nm)')
            axs[i + 1].set_ylabel(r'$a_{tot}\  (m^{-1})$')
            axs[i + 1].set_xlabel(r'Wavelength (nm)')
            axs[i + 2].set_ylabel(r'$b_{b_{tot}}\  (m^{-1})$')
            axs[i + 2].set_xlabel(r'Wavelength (nm)')
            # axs[i].plot(wl_cops, cops.loc[137,:].mes_avg,'grey')
            # axs[i].plot(wl_cops, cops.loc[136,:].mes_avg,'ok--')

        plt.suptitle(title + r' $bb_{phy}$ rescaled by ' + str(coef), fontsize=18)

        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.86])

        fig.savefig(os.path.join(figdir, 'Rrs_iop_chl_compar_bbphy_flat_rescale' + str(coef) + '.png'), dpi=300)
        plt.close(fig)
