"""
Plotting helpers for radiative transfer look-up tables, in particular polar
diagrams of radiance (or Stokes parameters) as a function of viewing zenith
angle and relative azimuth.
"""
import numpy as np
import cmocean as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties


class plot():
    """
    Polar plotting utilities.

    Instantiation sets the tick label size of matplotlib (``plt.rc``).

    :param font: font dictionary used for titles; if None, a serif font of
        size ``size`` is used
    :param size: font size of titles (used only if ``font`` is None)
    :param ticklabelsize: font size of the x and y tick labels
    """
    def __init__(self, font=None, size=20, ticklabelsize=12):
        # ----------------------------
        # set plotting styles

        plt.rc('xtick', labelsize=ticklabelsize)
        plt.rc('ytick', labelsize=ticklabelsize)

        if font == None:
            font = {'family': 'serif',
                    'color': 'black',
                    'weight': 'normal',
                    'size': size,
                    }
        self.font = font

    def label_polplot(self, ax, yticks=[20., 40., 60.], ylabels=[r'$20^{\circ}$', r'$40^{\circ}$', r'$60^{\circ}$']):
        """
        Set radial ticks and orientation of a polar axis: azimuth 0 at the
        top (North) and increasing clockwise.

        :param ax: matplotlib polar axis
        :param yticks: radial tick positions (deg)
        :param ylabels: radial tick labels
        """
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        return

    def add_polplot(self, ax, r, theta, values, title="", scale=True,
                    nlayers=25, cmap=cm.cm.delta, minmax=None, quantile=0.99,
                    colfmt='%0.1e', fontsize=12, pad=0.1, fraction=0.034, **kwargs):
        '''
        Plot polar diagram of raster expressed in radius and azimuth.

        :param ax: matplotlib polar axis
        :param r: 2D array of radius (can be VZA or SZA angles in deg)
        :param theta: 2D array of azimuth (rad)
        :param values: 2D raster to plot; if ``minmax`` is None it must
            provide ``min()`` and ``quantile()`` methods
            (e.g., :class:`xarray.DataArray`)
        :param title: title of the subplot
        :param scale: if True plot colorbar
        :param nlayers: number of contour levels
        :param cmap: colormap
        :param minmax: tuple ``(min, max)`` of the color scale; if None,
            the minimum and the ``quantile`` of ``values`` are used, unless
            ``vmin`` / ``vmax`` are given in ``kwargs``
        :param quantile: quantile of ``values`` used as maximum of the
            color scale when ``minmax`` is None
        :param colfmt: format of the colorbar labels
        :param fontsize: font size of the colorbar tick labels
        :param pad: padding between the axis and the colorbar
        :param fraction: fraction of the axis used for the colorbar
        :param kwargs: additional arguments passed to
            :meth:`matplotlib.axes.Axes.contourf`
        :return: the contour set (can be used to build a common colorbar)
        '''

        self.label_polplot(ax)
        ax.set_title(title, fontdict=self.font, pad=30)

        if minmax is not None:
            min_, max_ = minmax
        else:
            min_, max_ = values.min(), values.quantile(quantile)
            if 'vmin' in kwargs:
                min_ = kwargs['vmin']
            if 'vmax' in kwargs:
                max_ = kwargs['vmax']

            if max_ - min_ == 0:
                max_ = min_ + 1
        inc = (max_ - min_) / nlayers
        contour = np.arange(min_, max_, inc)
        cax = ax.contourf(theta, r, values, contour, extend='both', cmap=cmap, **kwargs)
        if scale:
            cbar = plt.colorbar(cax, ax=ax, format=colfmt, fraction=fraction, pad=pad)
            cbar.ax.tick_params(labelsize=fontsize)
        return cax
