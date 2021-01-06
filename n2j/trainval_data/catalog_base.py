import os
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pandas as pd
from n2j import trainval_data
import n2j.trainval_data.raytracing_utils as ru
import n2j.trainval_data.coord_utils as cu
from lenstronomy.LensModel.lens_model import LensModel


def get_catalog(catalog_name):
    return getattr(trainval_data, catalog_name.lower())


class CatalogBase(ABC):
    halo_cols_generic = ['halo_mass', 'stellar_mass', 'is_central',
                         'ra', 'dec', 'z']

    def __init__(self,
                 out_dir: str = '.',
                 test: bool = False,):
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.test = test
        # Paths for output data files
        self.sightlines_path = os.path.join(self.out_dir, 'sightlines.csv')
        self.halo_path_fmt = os.path.join(self.out_dir,
                                          'los_halos_los={:d}.csv')
        self.kappa_map_fmt = os.path.join(self.out_dir,
                                          'kappa_map_los={:d}.npy')
        self.gamma_map_fmt = os.path.join(self.out_dir,
                                          'gamma{:d}_map_los={:d}.npy')
        self.kappa_samples_fmt = os.path.join(self.out_dir,
                                              'kappa_samples_los={:d}.npy')
        self.uncalib_path = os.path.join(self.out_dir, 'uncalib.txt')  # FIXME
        open(self.uncalib_path, 'a').close()
        # Optionally overridden
        self.halo_satisfies = []

    @property
    @classmethod
    @abstractmethod
    def to_generic(cls):
        return NotImplemented

    @abstractmethod
    def get_generator(self):
        return NotImplemented

    @abstractmethod
    def rename_cols(self):
        return NotImplemented

    @abstractmethod
    def get_sightlines_on_grid(self):
        return NotImplemented

    def halo_conditions(self, halo_df: pd.DataFrame):
        """Conditions that halos should satisfy

        """
        stacked = np.array([cond(halo_df) for cond in self.halo_satisfies])
        return np.all(stacked, axis=0)

    def get_los_halos(self,
                      ra_los: float,
                      dec_los: float,
                      z_src: float,
                      fov: float,
                      out_path: str = None):
        halos = pd.DataFrame()  # neighboring galaxies in LOS
        # Iterate through chunks to bin galaxies into the partitions
        for df in self.get_generator(self.halo_cols):
            self.rename_cols(df)
            # Get galaxies in the aperture and in foreground of source
            # Discard smaller masses, since they won't have a big impact anyway
            lower_z = df['z'].values < z_src
            if lower_z.any():  # there are still some lower-z halos
                pass
            else:  # z started getting too high, no need to continue?
                continue
            include = np.logical_and(lower_z, self.halo_conditions(df))
            df = df[include].reset_index(drop=True)
            if len(df) > 0:
                ra_halo = df['ra'].values
                dec_halo = df['dec'].values
                d, ra_diff, dec_diff = cu.get_distance(ra_f=ra_halo,
                                                       dec_f=dec_halo,
                                                       ra_i=ra_los,
                                                       dec_i=dec_los)
                df['dist'] = d*60.0  # deg to arcmin
                df['ra_diff'] = ra_diff  # deg
                df['dec_diff'] = dec_diff  # deg
                halos = halos.append(df[df['dist'].values < fov*0.5],
                                     ignore_index=True)
            else:
                continue

        #####################
        # Define NFW kwargs #
        #####################
        halos['center_x'] = halos['ra_diff']*3600.0  # deg to arcsec
        halos['center_y'] = halos['dec_diff']*3600.0
        Rs, alpha_Rs, eff = ru.get_nfw_kwargs(halos['halo_mass'].values,
                                              halos['stellar_mass'].values,
                                              halos['z'].values,
                                              z_src,
                                              cosmo=self.cosmo)
        halos['Rs'] = Rs
        halos['alpha_Rs'] = alpha_Rs
        halos['eff'] = eff
        halos.reset_index(drop=True, inplace=True)
        self.rename_cols(halos)
        if out_path is not None:
            halos.to_csv(out_path, index=None)
        return halos

    def get_uncalib_kappa(self,
                          idx: int,
                          z_src: float,
                          fov: float,
                          diff: float,
                          n_samples: int,
                          halos: pd.DataFrame = None,
                          nfw_kwargs: List[dict] = None,
                          z_halos: List[float] = None,
                          map_kappa: bool = False,
                          map_gamma: bool = False,):
        """Raytrace through a single sightline

        """
        if halos is not None:
            z_halos = halos['z'].values
            nfw_params = ['Rs', 'alpha_Rs', 'center_x', 'center_y']
            nfw_kwargs = halos[nfw_params].to_dict('records')
        n_halos = len(nfw_kwargs)
        # Instantiate multi-plane lens model
        lens_model = LensModel(lens_model_list=['NFW']*n_halos,
                               z_source=z_src,
                               lens_redshift_list=z_halos,
                               multi_plane=True,
                               cosmo=self.cosmo,
                               observed_convention_index=[])
        uncalib_kappa = lens_model.kappa(0.0, 0.0, nfw_kwargs,
                                         diff=diff, diff_method='square')
        uncalib_g1, uncalib_g2 = lens_model.gamma(0.0, 0.0, nfw_kwargs,
                                                  diff=diff, diff_method='square')
        with open(self.uncalib_path, 'a') as f:
            f.write("{:d},\t{:f},\t{:f},\t{:f}\n".format(idx,
                                                         uncalib_kappa,
                                                         uncalib_g1,
                                                         uncalib_g2))
        if map_kappa:
            k_map = ru.get_kappa_map(lens_model, nfw_kwargs, fov,
                                     self.kappa_map_fmt.format(idx), diff=diff)
        else:
            k_map = None

        if map_gamma:
            g1_map, g2_map = ru.get_gamma_maps(lens_model, nfw_kwargs, fov,
                                               (self.gamma_map_fmt.format(1, idx),
                                                self.gamma_map_fmt.format(2, idx)),
                                               diff=diff)
        else:
            g1_map, g2_map = None, None

        if n_samples > 0:
            samples = self.sample_kappa(n_samples, halos=halos,
                                        lens_model=lens_model, fov=fov, diff=diff,
                                        out_path=self.kappa_samples_fmt.format(idx))
        else:
            samples = None

        out = dict(uncalib_k=uncalib_kappa,
                   uncalib_g1=uncalib_g1,
                   uncalib_g2=uncalib_g2,
                   k_map=k_map,
                   g1_map=g1_map,
                   g2_map=g2_map,
                   samples=samples)
        return out

    def sample_kappa(self,
                     n_samples: int,
                     fov: float,
                     diff: float,
                     halos: pd.DataFrame,
                     z_src: float = None,
                     lens_model: LensModel = None,
                     out_path: str = None):
        ################
        # Sample kappa #
        ################
        # gamma1, gamma2 are not resampled due to symmetry around 0
        kappa_samples = np.empty(n_samples)
        n_halos = halos.shape[0]
        if lens_model is None:
            lens_model = LensModel(lens_model_list=['NFW']*n_halos,
                                   z_source=z_src,
                                   lens_redshift_list=halos['z'].values,
                                   multi_plane=True,
                                   cosmo=self.cosmo,
                                   observed_convention_index=[])
        S = 0
        while S < n_samples:
            new_ra, new_dec = cu.sample_in_aperture(n_halos, fov*0.5/60.0)
            halos['center_x'] = new_ra*3600.0
            halos['center_y'] = new_dec*3600.0
            nfw_params = ['Rs', 'alpha_Rs', 'center_x', 'center_y']
            nfw_kwargs = halos[nfw_params].to_dict('records')
            resampled_kappa = lens_model.kappa(0.0, 0.0, nfw_kwargs, diff=diff)
            if resampled_kappa < 1.0:
                kappa_samples[S] = resampled_kappa
                S += 1
            else:  # halo fell on top of zeropoint, kappa too big!
                continue
        if out_path is not None:
            np.save(out_path, kappa_samples)
        return kappa_samples

    def raytrace_single(self,
                        idx: int,
                        **kwargs):
        los_info = self.sightlines.iloc[idx]
        # Get halos
        if os.path.exists(self.halo_path_fmt.format(idx)):
            halos = pd.read_csv(self.halo_path_fmt.format(idx), index_col=None)
        else:
            halos = self.get_los_halos(ra_los=los_info['ra'],
                                       dec_los=los_info['dec'],
                                       z_src=los_info['z'],
                                       fov=kwargs['fov'],
                                       out_path=self.halo_path_fmt.format(idx))
        # Get kappas
        if not os.path.exists(self.kappa_samples_fmt.format(idx)):
            _ = self.get_uncalib_kappa(idx, halos=halos, z_src=los_info['z'],
                                       **kwargs)

    def raytrace_all(self, n_sightlines: int, **kwargs):
        from tqdm import tqdm
        if not os.path.exists(self.sightlines_path):
            self.get_sightlines_on_grid(n_sightlines, kwargs['fov']*0.5/60.0)
        else:
            self._sightlines = pd.read_csv(self.sightlines_path, index_col=None)
        for i in tqdm(range(n_sightlines)):
            self.raytrace_single(i, **kwargs)