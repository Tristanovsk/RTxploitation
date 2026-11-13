"""
Atmospheric transmittance of the water-leaving signal from OSOAA simulations.

The OSOAA runs used here couple an atmosphere (Rayleigh + one lognormal
aerosol mode, AOT(550) = 0.1) with a sediment-loaded water body. Each
wavelength is simulated twice: with the ocean (``osoaa_tot_*``) and without
it (``osoaa_nosea_*``); their difference isolates the water-leaving
contribution to the Stokes vector at every level ``z``.

The script computes

1. the transmittance of the downward irradiance, ``Ed(0+) / Ed(TOA)``, split
   into its direct and diffuse parts;
2. the transmittance of the upward water-leaving radiance,
   ``Lw(TOA) / Lw(0+)``, for the total (I) and the polarized
   (sqrt(Q^2 + U^2)) radiance;
3. the combined (sun -> surface -> sensor) transmittance;
4. the fitted coefficients (a, b) of the classical approximation
   ``t = exp(-(a tau_r + b tau_a) / mu)``;
5. the total downward transmittance for the OPAC aerosol models of the
   GRS look-up table.

Run it from anywhere: ``python case_study/transmittance/stokes_modeling.py``.
The input paths can be changed with the ``RTX_SCATMAT_DIR`` and
``RTX_OPAC_LUT`` environment variables. The figures are written in
``case_study/transmittance/fig``.

A documented walk-through is available in
``case_study/transmittance/notebook/transmittance.ipynb``.
"""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

import RTxploitation.load_lut_nc as load

opj = os.path.join

# ---------------------------------
# Settings
# ---------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

SCATMAT_DIR = os.environ.get('RTX_SCATMAT_DIR', '/data/vrtc/lut/scat_mat/')
OPAC_LUT = os.environ.get(
    'RTX_OPAC_LUT',
    '/home/harmel/Dropbox/Dropbox/work/git/satellite_app/grs/grs/data/lut/opac_osoaa_lut_v3.nc')
RAYLEIGH_FILE = opj(ROOT, 'data', 'rayleigh_bodhaine.txt')
FIG_DIR = opj(HERE, 'fig')

# atmosphere common to all the scat_mat simulations
PREFIX = 'osoaa_tot_aot0.1_aero_rg0.10_sig0.46_nr1.45_ni-0.0010_ws2'


def set_style():
    """Serif fonts with STIX math; STIX ships with matplotlib (no missing-font warnings)."""
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix',
                         'font.size': 14, 'axes.labelsize': 15})


# ---------------------------------
# Loading
# ---------------------------------
def load_rayleigh(file=RAYLEIGH_FILE):
    """
    Rayleigh optical thickness of Bodhaine et al. (1999), 200-2450 nm.

    :return: :class:`xarray.DataArray` ``rot(wl)``, wl in nm
    """
    rot = pd.read_csv(file, sep=r'\s+', skiprows=16, header=None,
                      names=['wl', 'rot', 'dpol'])
    return rot.set_index('wl').to_xarray().rot


def simulation_labels(spm='1.00', nr='1.15', rmed='1.0', wls=np.arange(400, 851, 25)):
    """File name suffixes of the sediment simulations, one per wavelength (nm)."""
    return ['__sed%s_nr%s_rmed%s_wl%0.3f' % (spm, nr, rmed, wl / 1e3) for wl in wls]


def load_simulations(labels, idir=SCATMAT_DIR, prefix=PREFIX):
    """
    Fluxes and water-leaving Stokes parameters for the 3 last levels:
    ``z = -1e-6`` (0-, just below the surface), ``z = 0`` (0+) and TOA.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return load.load_osoaa(prefix=prefix, labels=labels, idir=idir, water_signal=True)


def load_optical_thickness(labels, idir=SCATMAT_DIR, prefix=PREFIX):
    """
    Rayleigh and aerosol optical properties actually used by OSOAA.

    :return: :class:`xarray.Dataset` along ``wavelength`` (nm) with ``rot``,
        ``aot``, ``ssa`` and ``asym_factor`` (aerosol asymmetry parameter)
    """
    out = []
    for label in labels:
        file = opj(idir, prefix + label + '.nc')
        if not os.path.exists(file):
            continue
        aero = xr.open_dataset(file, group='optical_properties/aerosol')
        with xr.open_dataset(file) as root:
            rot = float(root.attrs['AP.MOT'])
        out.append(xr.Dataset({'rot': rot,
                               'aot': float(aero.aot),
                               'ssa': float(aero.ssa),
                               'asym_factor': float(aero.asym_factor)},
                              coords={'wavelength': round(float(aero.wl) * 1000, 3)}))
        aero.close()
    return xr.concat(out, dim='wavelength')


# ---------------------------------
# Transmittances
# ---------------------------------
def _level(x, which):
    """Select 0+ (``'surf'``) or TOA (``'toa'``) and drop the z coordinate."""
    z = 0. if which == 'surf' else float(x.z.max())
    return x.sel(z=z).drop_vars('z')


def downward_transmittance(x):
    """
    Transmittance of the downward irradiance from TOA to 0+.

    At TOA the downward irradiance is purely direct, Ed(TOA) = mu0 F0, so

    * ``T_tot = Ed(0+) / Ed(TOA)``
    * ``T_dir = Eddir(0+) / Ed(TOA)`` (= exp(-tau / mu0))
    * ``T_dif = (Ed - Eddir)(0+) / Ed(TOA)``
    """
    surf, toa = _level(x, 'surf'), _level(x, 'toa')
    return xr.Dataset({'T_tot': surf.Ed / toa.Ed,
                       'T_dir': surf.Eddir / toa.Ed,
                       'T_dif': (surf.Ed - surf.Eddir) / toa.Ed})


def upward_transmittance(x):
    """
    Transmittance of the upward water-leaving radiance from 0+ to TOA.

    * ``t_I = I(TOA) / I(0+)``
    * ``t_P = Lp(TOA) / Lp(0+)`` with ``Lp = sqrt(Q^2 + U^2)``
    * ``DoLP_surf``, ``DoLP_toa``: degree of linear polarization Lp / I

    The ratios of Q and U themselves are not used: they are ill-defined
    where Q or U crosses zero (e.g. U in the principal plane).
    """
    surf, toa = _level(x, 'surf'), _level(x, 'toa')
    lp_surf = np.sqrt(surf.Q ** 2 + surf.U ** 2)
    lp_toa = np.sqrt(toa.Q ** 2 + toa.U ** 2)
    return xr.Dataset({'t_I': toa.I / surf.I,
                       't_P': lp_toa / lp_surf,
                       'DoLP_surf': lp_surf / surf.I,
                       'DoLP_toa': lp_toa / toa.I})


def transmittance_approx(rot, aot, mu, a=0.5, b=0.16):
    """Classical diffuse transmittance approximation exp(-(a tau_r + b tau_a) / mu)."""
    return np.exp(-(a * rot + b * aot) / mu)


def fit_coefficients(T, rot, aot, mu):
    """
    Least-squares fit of (a, b) in ``T = exp(-(a rot + b aot) / mu)`` along wavelength.

    ``-mu ln T = a rot + b aot`` is linear in (a, b); the fit is done for every
    remaining geometry of ``T``.

    :param T: transmittance with a ``wavelength`` dimension
    :param rot, aot: optical thicknesses along ``wavelength``
    :param mu: cosine of the zenith angle (broadcastable to ``T``)
    :return: :class:`xarray.Dataset` with ``a``, ``b`` and the rms residual of T
    """
    y = -mu * np.log(T)
    A = np.stack([rot.values, aot.values], axis=1)

    def _fit(yv):
        if not np.all(np.isfinite(yv)):
            return np.array([np.nan, np.nan, np.nan])
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
        return np.append(coef, np.sqrt(np.mean((np.exp(-yv) - np.exp(-A @ coef)) ** 2)))

    res = xr.apply_ufunc(_fit, y, input_core_dims=[['wavelength']], output_core_dims=[['coef']],
                         vectorize=True)
    return xr.Dataset({'a': res.isel(coef=0), 'b': res.isel(coef=1),
                       'rms': res.isel(coef=2)})


def mu(angle):
    return np.cos(np.radians(angle))


# ---------------------------------
# Plots
# ---------------------------------
def plot_irradiance(x, szas=(10, 30, 50, 70)):
    fig, axs = plt.subplots(ncols=2, figsize=(14, 5), sharex=True)
    surf = _level(x, 'surf')
    for i, sza in enumerate(szas):
        c = 'C%d' % i
        surf.Ed.sel(sza=sza).plot(ax=axs[0], color=c, label='%d' % sza)
        surf.Eddir.sel(sza=sza).plot(ax=axs[0], color=c, ls='--')
        surf.Eu.sel(sza=sza).plot(ax=axs[1], color=c, label='%d' % sza)
    axs[0].set_title('Downward irradiance at 0+ (dashed: direct)')
    axs[1].set_title('Upward irradiance at 0+')
    for ax in axs:
        ax.set_xlabel('Wavelength (nm)')
        ax.legend(title='SZA (deg)')
    fig.tight_layout()
    return fig


def plot_downward_transmittance(Td, tau, szas=(0, 30, 50, 70)):
    fig, axs = plt.subplots(ncols=3, figsize=(17, 5), sharex=True, sharey=True)
    for i, sza in enumerate(szas):
        c = 'C%d' % i
        m0 = mu(sza)
        Td.T_tot.sel(sza=sza).plot(ax=axs[0], color=c, label='%d' % sza)
        axs[0].plot(tau.wavelength, transmittance_approx(tau.rot, tau.aot, m0, a=0.5,
                                                         b=1 - tau.ssa * (1 + tau.asym_factor) / 2),
                    color=c, ls=':')
        Td.T_dir.sel(sza=sza).plot(ax=axs[1], color=c, label='%d' % sza)
        axs[1].plot(tau.wavelength, np.exp(-(tau.rot + tau.aot) / m0), color=c, ls=':')
        Td.T_dif.sel(sza=sza).plot(ax=axs[2], color=c, label='%d' % sza)
    axs[0].set_title('Total $E_d(0^+)/E_d(TOA)$')
    axs[1].set_title('Direct (dotted: $e^{-(\\tau_r+\\tau_a)/\\mu_0}$)')
    axs[2].set_title('Diffuse')
    for ax in axs:
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('')
        ax.legend(title='SZA (deg)', fontsize=10)
    axs[0].set_ylabel('Transmittance')
    fig.tight_layout()
    return fig


def plot_stokes_spectra(x, sza=30, vza_idx=6, azis=(0, 50, 100, 150, 180)):
    fig, axs = plt.subplots(ncols=3, figsize=(16, 5))
    surf = _level(x, 'surf').isel(vza=vza_idx).sel(sza=sza)
    for i, param in enumerate(['I', 'Q', 'U']):
        surf[param].sel(azi=list(azis)).plot(x='wavelength', hue='azi', ax=axs[i])
        axs[i].set_title('%s at 0+, SZA=%d, VZA=%.1f' % (param, sza, float(surf.vza)))
        axs[i].set_xlabel('Wavelength (nm)')
    fig.tight_layout()
    return fig


def plot_upward_transmittance_polar(tu, tau, szas=(10, 30, 60), vza_idx=8, azis=(0, 50, 100, 150, 200)):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 14), sharex=True)
    vza = float(tu.vza[vza_idx])
    mv = mu(vza)
    for r, sza in enumerate(szas):
        t = tu.isel(vza=vza_idx).sel(sza=sza, azi=list(azis))
        t.t_I.plot(x='wavelength', hue='azi', ax=axs[r, 0], add_legend=r == 0)
        axs[r, 0].plot(tau.wavelength, transmittance_approx(tau.rot, tau.aot, mv, 0.52, 0.14),
                       color='black', ls='--', label='approx.')
        t.t_P.plot(x='wavelength', hue='azi', ax=axs[r, 1], add_legend=False)
        axs[r, 1].plot(tau.wavelength, np.exp(-(tau.rot + tau.aot) / mv), color='grey', ls=':')
        for azi in azis:
            t.DoLP_surf.sel(azi=azi).plot(x='wavelength', ax=axs[r, 2])
        axs[r, 2].set_prop_cycle(None)
        for azi in azis:
            t.DoLP_toa.sel(azi=azi).plot(x='wavelength', ax=axs[r, 2], ls='--')
        axs[r, 0].set_ylim(0.6, 1.1)
        axs[r, 1].set_ylim(0.4, 1.3)
        titles = ['$t_I$', '$t_P$ (dotted: direct)', 'DoLP 0+ (solid), TOA (dashed)']
        for c in range(3):
            axs[r, c].set_title('%s, SZA=%d, VZA=%.1f' % (titles[c], sza, vza), fontsize=13)
            axs[r, c].set_ylabel('')
            axs[r, c].set_xlabel('')
    for c in range(3):
        axs[-1, c].set_xlabel('Wavelength (nm)')
    fig.tight_layout()
    return fig


def plot_transmittance_grid(T, tau, approx_fn, ylabel, vza_idxs=(1, 5, 11), szas=(10, 30, 60), azis=(0, 50, 100)):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), sharex=True, sharey=True)
    for r, iv in enumerate(vza_idxs):
        for c, sza in enumerate(szas):
            ax = axs[r, c]
            vza = float(T.vza[iv])
            ax.plot(tau.wavelength, approx_fn(mu(sza), mu(vza)), color='black', ls='--', label='approx.')
            for azi in azis:
                T_ = T.isel(vza=iv).sel(sza=sza, azi=azi)
                ax.plot(T_.wavelength, T_.values, label='%d' % azi)
            ax.set_title('VZA=%.1f, SZA=%d' % (vza, sza), fontsize=14)
            ax.legend(title='azimuth', fontsize=10)
            ax.minorticks_on()
    for i in range(3):
        axs[-1, i].set_xlabel('Wavelength (nm)')
        axs[i, 0].set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def upward_approx_error(tu, tau, a=0.52, b=0.16, vza_max=70):
    """
    Relative error of the approximation exp(-(a tau_r + b tau_a)/mu_v) on the
    azimuth-averaged transmittance t_I of the upward water-leaving radiance.
    """
    t_az = tu.t_I.sel(azi=slice(0, 355)).mean('azi').sel(vza=slice(0, vza_max))
    t_app = transmittance_approx(tau.rot, tau.aot, mu(t_az.vza), a, b)
    return t_app / t_az - 1


def plot_fitted_coefficients(fit_down, err_up, szas=(0, 30, 60), wls=(400, 550, 850)):
    fig, axs = plt.subplots(ncols=4, figsize=(20, 5))
    axs[0].plot(fit_down.sza, fit_down.a, '-o', label='a (Rayleigh)')
    axs[0].plot(fit_down.sza, fit_down.b, '-o', label='b (aerosol)')
    axs[0].set_title('Fit of $E_d(0^+)/E_d(TOA)$')
    axs[0].set_xlabel('SZA (deg)')
    axs[0].legend(fontsize=10)
    for i, sza in enumerate(szas):
        ax = axs[i + 1]
        for wl in wls:
            (100 * err_up.sel(sza=sza).sel(wavelength=wl, method='nearest')).plot(ax=ax, label='%d nm' % wl)
        ax.axhline(0, color='grey', lw=0.8)
        ax.set_title('Error of $t_I$ approx., SZA=%d' % sza)
        ax.set_xlabel('VZA (deg)')
        ax.set_ylabel('Relative error (%)' if i == 0 else '')
        ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def plot_opac_downward(lut_file=OPAC_LUT, sza=30, rh='_rh70',
                       models=('COAV', 'DESE', 'MACL', 'MAPO', 'COPO', 'URBA')):
    lut = xr.open_dataset(lut_file)
    lut['wl'] = lut['wl'] * 1000
    rot = load_rayleigh()
    wl = np.linspace(400, 2450, 500)
    rot_ = rot.interp(wl=wl)
    m0 = mu(sza)

    Ttot = (lut.Ed.sel(z=0) / lut.Ed.sel(z=lut.z.max())).sel(sza=sza)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        '', ['navy', 'blue', 'lightskyblue', 'grey', 'forestgreen', 'yellowgreen',
             'khaki', 'gold', 'orangered', 'firebrick', 'purple'])
    norm = mpl.colors.Normalize(vmin=0.001, vmax=1.5)

    fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharey=True, sharex=True)
    axs = axs.ravel()
    for i_, model_ in enumerate(models):
        model = model_ + rh
        lut_m = lut.sel(model=model)
        T = Ttot.sel(model=model).interp(wl=wl, method='cubic')
        aot = lut_m.aot.interp(wl=wl, method='cubic')
        ssa = lut_m.ssa.interp(wl=wl, method='cubic')
        # absorption-only approximation: scattered aerosol light is assumed to reach the surface
        Tapp = np.exp(-(0.52 * rot_ + aot * (1 - ssa)) / m0)
        for aot_ref in lut.aot_ref.values[1:]:
            color = cmap(norm(aot_ref))
            axs[i_].plot(wl, T.sel(aot_ref=aot_ref), color=color, lw=1.2, label='%g' % aot_ref)
            axs[i_].plot(wl, Tapp.sel(aot_ref=aot_ref), color=color, lw=1, ls=':')
        axs[i_].set_title(model)
        axs[i_].minorticks_on()
    axs[0].legend(title='AOT(550)', fontsize=10, ncol=2)
    for ax in axs[3:]:
        ax.set_xlabel('Wavelength (nm)')
    for ax in axs[::3]:
        ax.set_ylabel('$E_d(0^+)/E_d(TOA)$, SZA=%d' % sza)
    fig.tight_layout()
    lut.close()
    return fig


# ---------------------------------
# Main
# ---------------------------------
def main(show=False):
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    labels = simulation_labels()
    x = load_simulations(labels)
    tau = load_optical_thickness(labels)

    Td = downward_transmittance(x)
    tu = upward_transmittance(x)

    figs = {
        'irradiance.png': plot_irradiance(x),
        'downward_transmittance.png': plot_downward_transmittance(Td, tau),
        'stokes_spectra.png': plot_stokes_spectra(x),
        'upward_transmittance_polar.png': plot_upward_transmittance_polar(tu, tau),
        'upward_transmittance.png': plot_transmittance_grid(
            tu.t_I, tau, lambda m0, mv: transmittance_approx(tau.rot, tau.aot, mv, 0.52, 0.16),
            '$L_w$ transmittance $t_I$'),
        'combined_total_transmittance.png': plot_transmittance_grid(
            tu.t_I * Td.T_tot, tau,
            lambda m0, mv: np.exp(-(0.5 * tau.rot + 0.12 * tau.aot) * (1 / m0 + 1 / mv)),
            'Combined transmittance'),
    }

    # fit of the approximation coefficients (downward) and error of the classical one (upward)
    T_ = Td.T_tot.sel(sza=slice(0, 70))
    fit_down = fit_coefficients(T_, tau.rot, tau.aot, mu(T_.sza))
    err_up = upward_approx_error(tu, tau)
    figs['fitted_coefficients.png'] = plot_fitted_coefficients(fit_down, err_up)
    print('Downward irradiance fit:\n', fit_down.to_dataframe())

    if os.path.exists(OPAC_LUT):
        figs['opac_downward_transmittance.png'] = plot_opac_downward()
    else:
        print('OPAC LUT not found (%s): skipped' % OPAC_LUT)

    for name, fig in figs.items():
        fig.savefig(opj(FIG_DIR, name), dpi=200)
    if show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
