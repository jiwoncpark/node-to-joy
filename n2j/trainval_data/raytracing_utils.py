import numpy as np
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

__all__ = ['get_nfw_kwargs', 'get_kappa_map', 'get_gamma_maps']
__all__ += ['get_concentration']


def get_nfw_kwargs(halo_mass, stellar_mass, halo_z, z_src, cosmo):
    c_200 = get_concentration(halo_mass, stellar_mass)
    n_halos = len(c_200)
    Rs_all, alpha_Rs_all = np.empty(n_halos), np.empty(n_halos)
    lensing_eff_all = np.empty(n_halos)
    for h in range(n_halos):
        lc = LensCosmo(z_lens=halo_z[h], z_source=z_src, cosmo=cosmo)
        lensing_eff = lc.dds/lc.ds
        Rs_angle, alpha_Rs = lc.nfw_physical2angle(M=halo_mass[h],
                                                   c=c_200[h])
        Rs_all[h] = Rs_angle
        alpha_Rs_all[h] = alpha_Rs
        lensing_eff_all[h] = lensing_eff
    return Rs_all, alpha_Rs_all, lensing_eff_all


def get_kappa_map(lens_model, nfw_kwargs, fov, save_path, diff,
                  x_grid=None, y_grid=None):
    """Plot a map of kappa and save to disk

    """
    # 1 asec rez, in arcsec units
    if x_grid is None:
        x_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0
    if y_grid is None:
        y_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0
    xx, yy = np.meshgrid(x_grid, y_grid)
    kappa_map = lens_model.kappa(xx, yy, nfw_kwargs, diff=diff,
                                 diff_method='square')
    np.save(save_path, kappa_map)
    return kappa_map


def get_gamma_maps(lens_model, nfw_kwargs, fov, save_path, diff,
                   x_grid=None, y_grid=None):
    """Plot a map of gamma and save to disk

    """
    # 1 asec rez, in arcsec units
    if x_grid is None:
        x_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0
    if y_grid is None:
        y_grid = np.arange(-fov*0.5, fov*0.5, 1/60.0)*60.0
    xx, yy = np.meshgrid(x_grid, y_grid)
    gamma1_map, gamma2_map = lens_model.gamma(xx, yy, nfw_kwargs, diff=diff,
                                              diff_method='square')
    np.save(save_path[0], gamma1_map)
    np.save(save_path[1], gamma2_map)
    return gamma1_map, gamma2_map


def get_concentration(halo_mass, stellar_mass,
                      m=-0.10, A=3.44, trans_M_ratio=430.49, c_0=3.19):
    """Get the halo concentration from halo and stellar masses
    using the fit in Childs et al 2018 for all individual halos, both relaxed
    and unrelaxed

    Parameters
    ----------
    trans_M_ratio : float or np.array
        ratio of the transition mass to the stellar mass

    """
    halo_M_ratio = halo_mass/stellar_mass
    b = trans_M_ratio  # trans mass / stellar mass
    c_200 = A*(((halo_M_ratio/b)**m)*((1.0 + (halo_M_ratio/b))**(-m)) - 1.0) + c_0
    c_200 = np.maximum(c_200, 1.0)
    return c_200