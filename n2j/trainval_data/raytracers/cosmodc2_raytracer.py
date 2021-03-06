"""Raytracing module for CosmoDC2

"""
import os
import time
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from n2j import data
from n2j.trainval_data.raytracers.base_raytracer import BaseRaytracer
import n2j.trainval_data.cosmodc2_raytracing_utils as ru

__all__ = ['CosmoDC2Raytracer']


class CosmoDC2Raytracer(BaseRaytracer):
    """Raytracing tool for postprocessing the lensing distortion in CosmoDC2

    """
    NSIDE = 32
    LENSING_NSIDE = 4096

    def __init__(self, out_dir, fov, n_kappa_samples, healpix,
                 mass_cut=11, n_sightlines=1000, debug=False):
        """
        Parameters
        ----------
        out_dir : str or os.path
        fov : float
            field of view in arcmin
        mass_cut : float
            log10(minimum halo mass) (Default: 11.0)
        n_sightlines : int
            number of sightlines to raytrace through (Default: 1000)

        """
        np.random.seed(123)
        BaseRaytracer.__init__(self, out_dir, debug)
        self.fov = fov
        self.mass_cut = mass_cut
        self.n_sightlines = n_sightlines
        self.n_kappa_samples = n_kappa_samples
        self.healpix = healpix
        self._get_pointings()
        self.debug = debug
        logged_cols = ['idx', 'kappa', 'gamma1', 'gamma2', 'weighted_mass_sum']
        uncalib_df = pd.DataFrame(columns=logged_cols)
        uncalib_df.to_csv(self.uncalib_path, index=None)

    def get_pointings_iterator(self, healpix, columns, chunksize=100000):
        """Get an iterator over the galaxy catalog defining the pointings

        """
        cat_path = os.path.join(data.__path__[0],
                                'cosmodc2_{:d}'.format(healpix), 'raw',
                                'cosmodc2_pointings_{:d}.csv'.format(healpix))
        if self.debug:
            cat = pd.read_csv(cat_path, chunksize=50, nrows=1000,
                              usecols=columns)
            # Include z~2 galaxies
            cat['redshift'] = np.maximum(0.1, 2.0 + np.random.randn(100))
        else:
            cat = pd.read_csv(cat_path, chunksize=chunksize, nrows=None,
                              usecols=columns)
        return cat

    def get_halos_iterator(self, healpix, columns, chunksize=100000):
        """Get an iterator over the halo catalog defining our line-of-sight
        halos

        """
        cat_path = os.path.join(data.__path__[0],
                                'cosmodc2_{:d}'.format(healpix), 'raw',
                                'cosmodc2_halos_{:d}.csv'.format(healpix))
        if self.debug:
            cat = pd.read_csv(cat_path, chunksize=50, nrows=1000,
                              usecols=columns)
        else:
            cat = pd.read_csv(cat_path, chunksize=chunksize, nrows=None,
                              usecols=columns)
        return cat

    def _get_pointings(self):
        """Gather pointings defining our sightlines

        """
        if os.path.exists(self.sightlines_path):
            self.pointings = pd.read_csv(self.sightlines_path,
                                         index_col=None,
                                         nrows=self.n_sightlines)
        else:
            start = time.time()
            sightline_cols = ['ra_true', 'dec_true', 'redshift_true']
            sightline_cols += ['convergence', 'shear1', 'shear2', 'galaxy_id']
            gals_cat = self.get_pointings_iterator(self.healpix, sightline_cols)
            self.pointings = ru.get_pointings_on_grid(self.healpix,
                                                      gals_cat,
                                                      self.n_sightlines,
                                                      self.sightlines_path,
                                                      self.fov*0.5/60.0,
                                                      nside=self.NSIDE)
            end = time.time()
            print("Generated {:d} sightline(s) in"
                  " {:.2f} min.".format(self.n_sightlines, (end-start)/60.0))

    def single_raytrace(self, i):
        """Raytrace through a single sightline

        """
        sightline = self.pointings.iloc[i]
        halo_cols = ['halo_mass', 'stellar_mass']
        halo_cols += ['ra_true', 'dec_true', 'redshift_true']
        halos_cat = self.get_halos_iterator(self.healpix, halo_cols)
        ru.raytrace_single_sightline(i,
                                     halos_cat,
                                     ra_los=sightline['ra'],
                                     dec_los=sightline['dec'],
                                     z_src=sightline['z'],
                                     fov=self.fov,
                                     map_kappa=self.debug,
                                     map_gamma=self.debug,
                                     n_kappa_samples=self.n_kappa_samples,
                                     mass_cut=self.mass_cut,
                                     out_dir=self.out_dir,
                                     test=self.debug)

    def parallel_raytrace(self):
        """Raytrace through multiple sightlines in parallel

        """
        n_cores = min(multiprocessing.cpu_count() - 1, self.n_sightlines)
        with multiprocessing.Pool(n_cores) as pool:
            return list(tqdm(pool.imap(self.single_raytrace,
                                       range(self.n_sightlines)),
                             total=self.n_sightlines))

