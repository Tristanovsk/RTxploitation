import numpy as np
import cmocean as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties


class plot():
    def __init__(self, font=None):
        # ----------------------------
        # set plotting styles

        plt.rcParams.update({'font.size': 16})

        if font == None:
            font = {'family': 'serif',
                    'color': 'black',
                    'weight': 'normal',
                    'size': 22,
                    }
        self.font = font

    def label_polplot(self, ax, yticks=[20., 40., 60.], ylabels=['$20^{\circ}$', '$40^{\circ}$', '$60^{\circ}$']):

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        return

    def add_polplot(self, ax, r, theta, values, title="", scale=True, nlayers=25, cmap=cm.cm.delta, minmax=None, colfmt='%0.1e', pad=0.1, fraction=0.034,**kwargs):

        self.label_polplot(ax)
        ax.set_title(title, fontdict=self.font, pad=30)

        if minmax is not None:
            min_, max_ = minmax
        else:
            min_, max_ = values.min(), values.max()
            if 'vmin' in kwargs:
                min_ = kwargs['vmin']
            if 'vmax' in kwargs:
                max_ = kwargs['vmax']

            if max_ - min_ == 0:
                max_ = min_ + 1
        inc = (max_ - min_) / nlayers
        contour = np.arange(min_, max_, inc)
        cax = ax.contourf(theta, r, values, contour, extend='both', cmap=cmap,**kwargs)
        if scale:
            plt.colorbar(cax, ax=ax, format=colfmt, fraction=fraction, pad=pad)

        return cax