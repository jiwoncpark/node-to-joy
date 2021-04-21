"""Raytracing module for CosmoDC2

"""
import os
import time
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.cosmology import WMAP7   # WMAP 7-year cosmology
from lenstronomy.LensModel.lens_model import LensModel
from n2j.trainval_data.raytracers.base_raytracer import BaseRaytracer
import n2j.trainval_data.utils.coord_utils as cu
import n2j.trainval_data.utils.halo_utils as hu

__all__ = ['CosmoDC2Raytracer']

column_naming = {'ra_true': 'ra', 'dec_true': 'dec', 'redshift_true': 'z',
                 'convergence': 'kappa',
                 'shear1': 'gamma1', 'shear2': 'gamma2'}


class CosmoDC2Raytracer(BaseRaytracer):
    """Raytracing tool for postprocessing the lensing distortion in CosmoDC2

    """
    NSIDE = 32
    LENSING_NSIDE = 4096
    KAPPA_DIFF = 1.0  # arcsec
    COLUMN_NAMING = column_naming
    TO_200C = 0.85  # multiply FOF (b=0.168) masses by this to get 200c masses

    def __init__(self, in_dir, out_dir, fov, n_kappa_samples, healpix,
                 approx_z_src=2.0, mass_cut=11, n_sightlines=1000,
                 kappa_sampling_dir=None, debug=False):
        """
        Parameters
        ----------
        in_dir : str or os.path
            where input data are stored, i.e. the parent folder of `raw`
        out_dir : str or os.path
            where Y labels will be stored
        fov : float
            field of view in arcmin
        healpix : int
            healpix ID that will be supersampled
        approx_z_src : float
            approximate redshift of all sources, aka sightline galaxies
            (Default: 2.0)
        mass_cut : float
            log10(minimum halo mass) (Default: 11.0)
        n_sightlines : int
            number of sightlines to raytrace through (Default: 1000)

        """
        self.seed = 123
        self.rng = np.random.default_rng(seed=self.seed)
        BaseRaytracer.__init__(self, in_dir, out_dir, debug)
        self.fov = fov
        self.mass_cut = mass_cut
        self.approx_z_src = approx_z_src
        self.n_sightlines = n_sightlines
        self.n_kappa_samples = n_kappa_samples
        self.healpix = healpix
        if self.n_kappa_samples:  # kappa explicitly sampled
            self.kappa_sampling_dir = None
        else:  # kappa interpolated
            if os.path.exists(kappa_sampling_dir):
                self.kappa_sampling_dir = kappa_sampling_dir
            else:
                raise OSError("If kappas were not sampled for each sightline,"
                              " you must generate some pairs of weighted sum of"
                              " masses and mean of kappas and provide the"
                              " out_dir of that run.")
        self.debug = debug
        self._set_column_names()
        self._get_pointings()
        uncalib_df = pd.DataFrame(columns=self.uncalib_cols)
        uncalib_df.to_csv(self.uncalib_path, index=None)

    def _set_column_names(self):
        """Set column names to be stored

        """
        pointings_cols = ['kappa', 'gamma1', 'gamma2']
        pointings_cols += ['galaxy_id', 'ra', 'dec', 'z', 'eps']
        pointings_cols.sort()
        self.pointings_cols = pointings_cols
        halos_cols = ['ra', 'ra_diff', 'dec', 'dec_diff', 'z', 'dist']
        halos_cols += ['eff', 'halo_mass', 'stellar_mass', 'Rs', 'alpha_Rs']
        halos_cols += ['galaxy_id']
        halos_cols.sort()
        self.halos_cols = halos_cols
        uncalib_cols = ['idx', 'kappa', 'gamma1', 'gamma2']
        uncalib_cols += ['weighted_mass_sum']
        self.uncalib_cols = uncalib_cols
        Y_cols = ['final_kappa', 'final_gamma1', 'final_gamma2', 'mean_kappa']
        Y_cols += ['galaxy_id', 'ra', 'dec', 'z']
        self.Y_cols = Y_cols

    def get_pointings_iterator(self, columns=None, chunksize=100000):
        """Get an iterator over the galaxy catalog defining the pointings

        """
        if columns is None:
            columns = ['ra_true', 'dec_true', 'redshift_true']
            columns += ['convergence', 'shear1', 'shear2', 'galaxy_id']
        cat_path = os.path.join(self.in_dir,
                                'cosmodc2_{:d}'.format(self.healpix), 'raw',
                                'pointings_{:d}.csv'.format(self.healpix))
        if self.debug:
            cat = pd.read_csv(cat_path, chunksize=50, nrows=1000,
                              usecols=columns)
            # Include z~2 galaxies
            fake_z = np.maximum(0.1, self.approx_z_src + np.random.randn(100))
            cat['redshift'] = fake_z
        else:
            cat = pd.read_csv(cat_path, chunksize=chunksize, nrows=None,
                              usecols=columns)
        return cat

    def get_halos_iterator(self, columns=None, chunksize=100000):
        """Get an iterator over the halo catalog defining our line-of-sight
        halos

        """
        if columns is None:
            halos_cols = ['halo_mass', 'stellar_mass']
            halos_cols += ['ra_true', 'dec_true', 'redshift_true', 'galaxy_id']
        cat_path = os.path.join(self.in_dir,
                                'cosmodc2_{:d}'.format(self.healpix), 'raw',
                                'halos_{:d}.csv'.format(self.healpix))
        if self.debug:
            cat = pd.read_csv(cat_path, chunksize=50, nrows=1000,
                              usecols=halos_cols)
        else:
            cat = pd.read_csv(cat_path, chunksize=chunksize, nrows=None,
                              usecols=halos_cols)
        return cat

    def _get_pointings(self):
        """Gather pointings defining our sightlines

        """
        if os.path.exists(self.sightlines_path):
            pointings_arr = np.load(self.sightlines_path)
            pointings = pd.DataFrame(pointings_arr, columns=self.pointings_cols)
            self.pointings = pointings
        else:
            start = time.time()
            self.pointings = self._get_pointings_on_grid(self.fov*0.5/60.0)
            end = time.time()
            print("Generated {:d} sightline(s) in"
                  " {:.2f} min.".format(self.n_sightlines, (end-start)/60.0))

    def _get_pointings_on_grid(self, dist_thres):
        """Get the pointings on a grid of healpixes

        Parameters
        ----------
        dist_thres : float
            matching threshold between gridpoints and halo positions, in deg

        Notes
        -----
        Currently takes 1.2 min for 1000 sightlines.
        Doesn't have to be so rigorous about finding sightlines closest to grid.
        Two requirements are that sightlines need to be dominated by cosmic
        variance (span a few degrees) and that each sightline has a galaxy.

        """

        # Get centroids of D partitions by gridding the sky area and querying a
        # galaxy closest to each grid center at redshift z > self.approx_z_src
        # Each partition, centered at that galaxy, corresponds to an LOS
        target_nside = cu.get_target_nside(self.n_sightlines,
                                           nside_in=self.NSIDE)
        sightline_ids = cu.upgrade_healpix(self.healpix, False,
                                           self.NSIDE, target_nside)
        ra_grid, dec_grid = cu.get_healpix_centers(sightline_ids, target_nside,
                                                   nest=True)
        # Randomly choose number of sightlines requested
        rand_i = self.rng.choice(np.arange(len(ra_grid)),
                                 size=self.n_sightlines,
                                 replace=False)
        ra_grid, dec_grid = ra_grid[rand_i], dec_grid[rand_i]
        close_enough = np.zeros_like(ra_grid).astype(bool)  # all gridpoints False
        iterator = self.get_pointings_iterator()
        sightlines = pd.DataFrame()
        for df in iterator:
            high_z = df[(df['redshift_true'] > self.approx_z_src)].reset_index(drop=True)
            if len(high_z) > 0:
                remaining = ~close_enough
                passing, i_cat, dist = cu.match(ra_grid[remaining],
                                                dec_grid[remaining],
                                                high_z['ra_true'].values,
                                                high_z['dec_true'].values,
                                                dist_thres
                                                )
                more_sightlines = high_z.iloc[i_cat].copy()
                more_sightlines['eps'] = dist
                sightlines = sightlines.append(more_sightlines,
                                               ignore_index=True)
                close_enough[remaining] = passing
            if np.all(close_enough):
                break
        sightlines.reset_index(drop=True, inplace=True)
        sightlines.rename(columns=self.COLUMN_NAMING, inplace=True)
        sightlines.sort_index(axis=1, inplace=True)
        np.save(self.sightlines_path, sightlines.values)
        return sightlines

    def get_los_halos(self, i, ra_los, dec_los, z_src, galaxy_id_los):
        """Compile halos in the line of sight of a given galaxy

        """
        iterator = self.get_halos_iterator()
        # Sorted list of stored halo properties
        if os.path.exists(self.halo_path_fmt.format(i, galaxy_id_los)):
            halos_arr = np.load(self.halo_path_fmt.format(i, galaxy_id_los))
            halos = pd.DataFrame(halos_arr, columns=self.halos_cols)
            return halos
        halos = pd.DataFrame()  # neighboring galaxies in LOS
        # Iterate through chunks to bin galaxies into the partitions
        for df in iterator:
            # Get galaxies in the aperture and in foreground of source
            # Discard smaller masses, since they won't have a big impact anyway
            lower_z = df['redshift_true'].values + 1.e-7 < z_src
            if lower_z.any():  # there are still some lower-z halos
                pass
            else:  # z started getting too high, no need to continue
                continue
            high_mass = df['halo_mass'].values*self.TO_200C > 10.0**self.mass_cut
            cut = np.logical_and(high_mass, lower_z)
            df = df[cut].reset_index(drop=True)
            if len(df) > 0:
                d, ra_diff, dec_diff = cu.get_distance(ra_f=df['ra_true'].values,
                                                       dec_f=df['dec_true'].values,
                                                       ra_i=ra_los,
                                                       dec_i=dec_los
                                                       )
                df['dist'] = d*60.0  # deg to arcmin
                df['ra_diff'] = ra_diff  # deg
                df['dec_diff'] = dec_diff  # deg
                halos = halos.append(df[df['dist'].values < self.fov*0.5],
                                     ignore_index=True)
            else:
                continue

        #####################
        # Define NFW kwargs #
        #####################
        halos['halo_mass'] *= self.TO_200C
        Rs, alpha_Rs, eff = hu.get_nfw_kwargs(halos['halo_mass'].values,
                                              halos['stellar_mass'].values,
                                              halos['redshift_true'].values,
                                              z_src,
                                              seed=i)
        halos['Rs'] = Rs
        halos['alpha_Rs'] = alpha_Rs
        halos['eff'] = eff
        halos.reset_index(drop=True, inplace=True)
        halos.rename(columns=self.COLUMN_NAMING, inplace=True)
        halos.sort_index(axis=1, inplace=True)
        np.save(self.halo_path_fmt.format(i, galaxy_id_los), halos.values)
        return halos

    def single_raytrace(self, i):
        """Raytrace through a single sightline

        """
        sightline = self.pointings.iloc[i]
        halos = self.get_los_halos(i,
                                   sightline['ra'], sightline['dec'],
                                   sightline['z'], int(sightline['galaxy_id']))
        n_halos = halos.shape[0]
        # Instantiate multi-plane lens model
        lens_model = LensModel(lens_model_list=['NFW']*n_halos,
                               z_source=sightline['z'],
                               lens_redshift_list=halos['z'].values,
                               multi_plane=True,
                               cosmo=WMAP7,
                               observed_convention_index=[])
        halos['center_x'] = halos['ra_diff']*3600.0  # deg to arcsec
        halos['center_y'] = halos['dec_diff']*3600.0
        nfw_kwargs = halos[['Rs', 'alpha_Rs', 'center_x', 'center_y']].to_dict('records')
        uncalib_kappa = lens_model.kappa(0.0, 0.0, nfw_kwargs,
                                         diff=self.KAPPA_DIFF,
                                         diff_method='square')
        uncalib_gamma1, uncalib_gamma2 = lens_model.gamma(0.0, 0.0, nfw_kwargs,
                                                          diff=self.KAPPA_DIFF,
                                                          diff_method='square')
        # Log the uncalibrated shear/convergence and the weighted sum of halo masses
        w_mass_sum = np.log10(np.sum(halos['eff'].values*halos['halo_mass'].values))
        new_row_data = dict(idx=[i],
                            kappa=[uncalib_kappa],
                            gamma1=[uncalib_gamma1],
                            gamma2=[uncalib_gamma2],
                            weighted_mass_sum=[w_mass_sum],
                            )
        new_row = pd.DataFrame(new_row_data)
        new_row.to_csv(self.uncalib_path, index=None, mode='a', header=None)
        # Optionally map the uncalibrated shear and convergence on a grid
        if self.debug:
            hu.get_kappa_map(lens_model, nfw_kwargs, self.fov,
                             self.k_map_fmt.format(i),
                             self.KAPPA_DIFF)
            hu.get_gamma_maps(lens_model, nfw_kwargs, self.fov,
                              (self.g1_map_fmt.format(i),
                               self.g2_map_fmt.format(i)),
                              self.KAPPA_DIFF)
        if self.n_kappa_samples > 0:
            self.sample_kappas(i, lens_model, halos)

    def sample_kappas(self, i, lens_model, halos):

        """Render the halos in uniformly random locations within the aperture to
        sample the kappas. The mean of sampled kappas will be used as an estimate of
        the additional average kappa contribution of our halos

        """
        n_halos = halos.shape[0]
        # gamma1, gamma2 are not resampled due to symmetry around 0
        if os.path.exists(self.k_samples_fmt.format(i)):
            return None
        kappa_samples = np.empty(self.n_kappa_samples)
        S = 0
        while S < self.n_kappa_samples:
            new_ra, new_dec = cu.sample_in_aperture(n_halos, self.fov*0.5/60.0)
            halos['center_x'] = new_ra*3600.0
            halos['center_y'] = new_dec*3600.0
            nfw_kwargs = halos[['Rs', 'alpha_Rs', 'center_x', 'center_y']].to_dict('records')
            resampled_kappa = lens_model.kappa(0.0, 0.0, nfw_kwargs,
                                               diff=self.KAPPA_DIFF,
                                               diff_method='square')
            if resampled_kappa < 1.0:
                kappa_samples[S] = resampled_kappa
                if self.debug:
                    hu.get_kappa_map(lens_model, nfw_kwargs, self.fov,
                                     self.k_samples_map_fmt.format(i, S))
                S += 1
            else:  # halo fell on top of zeropoint!
                continue
        np.save(self.k_samples_fmt.format(i), kappa_samples)

    def parallel_raytrace(self):
        """Raytrace through multiple sightlines in parallel

        """
        n_cores = min(multiprocessing.cpu_count() - 1, self.n_sightlines)
        with multiprocessing.Pool(n_cores) as pool:
            return list(tqdm(pool.imap(self.single_raytrace,
                                       range(self.n_sightlines)),
                             total=self.n_sightlines))
