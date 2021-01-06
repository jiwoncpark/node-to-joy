import os
import time
from typing import List
import numpy as np
import pandas as pd
from astropy.cosmology import WMAP7
from n2j import data
from n2j.trainval_data.catalog_base import CatalogBase
import n2j.trainval_data.coord_utils as cu


class CosmoDC2(CatalogBase):
    cosmo = WMAP7  # WMAP 7-year cosmology
    nside = 32
    valid_healpix = [10450]
    to_generic = {'ra_true': 'ra', 'dec_true': 'dec',
                  'shear1': 'gamma1', 'shear2': 'gamma2',
                  'convergence': 'wl_kappa',
                  'redshift_true': 'z',
                  'halo_mass': 'halo_mass', 'stellar_mass': 'stellar_mass',
                  'is_central': 'is_central'}

    def __init__(self,
                 out_dir: str = '.',
                 test: bool = False,
                 healpix: int = None,):
        CatalogBase.__init__(self, out_dir, test)
        # Ideally class attributes
        self.from_generic = {v: k for k, v in self.to_generic.items()}
        self.halo_cols = [self.from_generic[c] for c in CatalogBase.halo_cols_generic]
        # Modifiable attributes
        self._healpix = healpix
        self.halo_satisfies = [lambda x: x['is_central'].values.astype(bool)]
        self.halo_satisfies += [lambda x: np.log10(x['halo_mass'].values) > 11]
        self._sightlines = None  # init
        self.test_data_path = os.path.join(data.__path__[0], 'test.csv')

    @property
    def healpix(self):
        return self._healpix

    @healpix.setter
    def healpix(self, hp):
        if hp in self.valid_healpix:
            self._healpix = hp
        else:
            raise ValueError("CosmoDC2 does not contain provided healpix.")

    @property
    def sightlines(self):
        if self._sightlines is None:
            if os.path.exists(self.sightlines_path):
                self._sightlines = pd.read_csv(self.sightlines_path,
                                               index_col=None)
                return self._sightlines
            else:
                raise ValueError("Must generate sightlines first.")
        else:
            return self._sightlines

    def get_generator(self,
                      columns: List[str] = None,
                      chunksize: int = 100000):
        """Get a generator of cosmoDC2, too big to store in memory at once
        Parameters
        ----------
        columns : list of columns to load. Must match the CSV header.
        chunksize : number of rows in each chunk

        """
        if self.test:
            cosmodc2 = pd.read_csv(self.test_data_path, chunksize=5, nrows=None,
                                   usecols=columns)
        else:
            p = os.path.join(data.__path__[0],
                             'cosmodc2_train', 'raw',
                             'cosmodc2_trainval_{:d}.csv'.format(self._healpix))
            cosmodc2 = pd.read_csv(p, chunksize=chunksize, nrows=None,
                                   usecols=columns)
        return cosmodc2

    def rename_cols(self, df):
        """Rename cosmoDC2-specific columns to more general ones

        """
        df.rename(columns=self.to_generic, inplace=True)

    def get_grid(self, n_sightlines: int):
        # Oversample healpix IDs
        target_nside = cu.get_target_nside(n_sightlines, self.nside)
        hp_ids = cu.upgrade_healpix(self._healpix, False,
                                    self.nside, target_nside)
        ra_grid, dec_grid = cu.get_healpix_centers(hp_ids,
                                                   target_nside, nest=True)
        # Randomly choose number of sightlines requested
        rand_i = np.random.choice(np.arange(len(ra_grid)),
                                  size=n_sightlines, replace=False)
        ra_grid, dec_grid = ra_grid[rand_i], dec_grid[rand_i]
        return ra_grid, dec_grid

    def get_sightlines_on_grid(self, n_sightlines: int, dist_thres: float):
        """Get the sightlines

        Parameters
        ----------
        n_sightlines: desired number of sightlines
        dist_thres: matching threshold between gridpoints and halo positions,
                    in deg

        Notes
        -----
        Currently takes 1.2 min for 1000 sightlines.
        Doesn't have to be so rigorous about finding sightlines closest to grid.
        Two requirements are that sightlines need to be dominated by cosmic variance
        (span a few degrees) and that each sightline has a galaxy.

        """

        # Get centroids of D partitions by gridding the sky area and querying a
        # galaxy closest to each grid center at redshift z > 2
        # Each partition, centered at that galaxy,
        # corresponds to a line of sight (LOS)
        start = time.time()
        ra_grid, dec_grid = self.get_grid(n_sightlines)
        close_enough = np.zeros_like(ra_grid).astype(bool)  # init
        sightline_cols = ['ra_true', 'dec_true', 'redshift_true']
        sightline_cols += ['convergence', 'shear1', 'shear2']
        cosmodc2 = self.get_generator(sightline_cols)
        sightlines = pd.DataFrame()
        for df in cosmodc2:
            high_z = df[(df['redshift_true'] > 2.0)].reset_index(drop=True)
            if len(high_z) > 0:
                remaining = ~close_enough
                passing, i_cat, dist = cu.match(ra_grid[remaining],
                                                dec_grid[remaining],
                                                high_z['ra_true'].values,
                                                high_z['dec_true'].values,
                                                dist_thres)
                more_sightlines = high_z.iloc[i_cat].copy()
                more_sightlines['eps'] = dist
                sightlines = sightlines.append(more_sightlines,
                                               ignore_index=True)
                close_enough[remaining] = passing
            if np.all(close_enough):
                break
        sightlines.reset_index(drop=True, inplace=True)
        self.rename_cols(sightlines)
        sightlines.to_csv(self.sightlines_path, index=None)
        end = time.time()
        print("Generated {:d} sightlines"
              " in {:.2f} min.".format(n_sightlines, (end-start)/60.0))
        self._sightlines = sightlines
        return sightlines