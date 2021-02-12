"""Script to raytrace through cosmoD2 sightlines

Example
-------
To run this script, pass in the destination directory as the argument::

    $ python n2j/raytrace_cosmodc2.py <dest_dir>

"""

import os
import argparse
import functools
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from n2j.trainval_data.cosmodc2_raytracing import (raytrace_single_sightline,
                                                   get_sightlines_on_grid)


def single_raytrace(i, healpix, sightlines, fov, map_kappa, map_gamma,
                    n_kappa_samples, mass_cut, dest_dir):
    """Wrapper around `raytrace_single_sightline` to enable multiprocessing"""
    sightline = sightlines.iloc[i]
    raytrace_single_sightline(i,
                              healpix,
                              sightline['ra'], sightline['dec'],
                              sightline['z'],
                              fov,
                              map_kappa,
                              map_gamma,
                              n_kappa_samples,
                              mass_cut,
                              dest_dir)


class Sightlines:
    """Set of sightlines in a cosmoDC2 field"""
    def __init__(self, dest_dir, fov, map_kappa, map_gamma,
                 mass_cut=11, n_sightlines=1000):
        """
        Parameters
        ----------
        dest_dir : str or os.path
        fov : float
            field of view in arcmin
        map_kappa : bool
            whether to generate grid maps of kappa
        mass_cut : float
            log10(minimum halo mass) (Default: 11.0)
        n_sightlines : int
            number of sightlines to raytrace through (Default: 1000)

        """
        np.random.seed(123)
        self.dest_dir = dest_dir
        if not os.path.exists(self.dest_dir):
            os.mkdir(self.dest_dir)
        self.fov = fov
        self.map_kappa = map_kappa
        self.map_gamma = map_gamma
        self.mass_cut = mass_cut
        self.n_sightlines = n_sightlines
        self.healpix = 10450
        self.sightlines_path = os.path.join(self.dest_dir, 'sightlines.csv')
        self._get_pointings()
        self.uncalib_path = os.path.join(self.dest_dir, 'uncalib.csv')
        uncalib_df = pd.DataFrame(columns=['idx', 'kappa', 'gamma1', 'gamma2'])
        uncalib_df.to_csv(self.uncalib_path, index=None)

    def _get_pointings(self):
        """Gather pointings defining our sightlines

        """
        if os.path.exists(self.sightlines_path):
            self.pointings = pd.read_csv(self.sightlines_path,
                                         index_col=None,
                                         nrows=self.n_sightlines)
        else:
            self.pointings = get_sightlines_on_grid(self.healpix,
                                                    self.n_sightlines,
                                                    self.sightlines_path,
                                                    self.fov*0.5/60.0)

    def parallel_raytrace(self):
        """Raytrace through multiple sightlines in parallel_raytrace

        """
        single = functools.partial(single_raytrace,
                                   healpix=self.healpix,
                                   sightlines=self.pointings,
                                   fov=self.fov,
                                   map_kappa=self.map_kappa,
                                   map_gamma=self.map_gamma,
                                   n_kappa_samples=1000,
                                   mass_cut=self.mass_cut,
                                   dest_dir=self.dest_dir)
        return list(tqdm(pool.imap(single, range(self.n_sightlines)),
                         total=self.n_sightlines))

    def apply_calibration(self):
        """Subtract off the extra mass added when we raytraced through
        parameterized halos

        """
        sightlines = pd.read_csv(self.sightlines_path, index_col=None)
        uncalib = pd.read_csv(self.uncalib_path, index_col=None)
        uncalib.drop_duplicates('idx', inplace=True)
        n_sightlines = sightlines.shape[0]
        mean_kappas = np.empty(n_sightlines)
        # Compute mean kappa of halos in each sightline
        for los_i in range(n_sightlines):
            samples_path = os.path.join(self.dest_dir,
                                        'k_samples_los={:d}.npy'.format(los_i))
            samples = np.load(samples_path)
            samples = samples[samples < 0.5]
            mean_kappas[los_i] = np.mean(samples)
        # To the WL quantities, add our raytracing and subtract mean mass
        final_kappas = uncalib['kappa'] + sightlines['kappa'] - mean_kappas
        final_gamma1 = uncalib['gamma1'] + sightlines['gamma1']
        final_gamma2 = uncalib['gamma2'] + sightlines['gamma2']
        sightlines['final_kappa'] = final_kappas
        sightlines['final_gamma1'] = final_gamma1
        sightlines['final_gamma2'] = final_gamma2
        # Update the sightlines df
        sightlines.to_csv(self.sightlines_path, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dest_dir',
                        help='destination folder for the kappa maps, samples')
    parser.add_argument('--fov', default=0.85, dest='fov', type=float,
                        help='field of view in arcmin (Default: 0.85)')
    parser.add_argument('--map_kappa', default=False, dest='map_kappa', type=bool,
                        help='whether to generate grid maps of kappa (Default: False)')
    parser.add_argument('--map_gamma', default=False, dest='map_gamma', type=bool,
                        help='whether to generate grid maps of gamma (Default: False)')
    parser.add_argument('--n_sightlines', default=1000, dest='n_sightlines', type=int,
                        help='number of sightlines to raytrace through (Default: 1000)')
    parser.add_argument('--mass_cut', default=10.5, dest='mass_cut', type=float,
                        help='log10(minimum halo mass/solar) (Default: 10.5)')
    args = parser.parse_args()

    n_cores = min(multiprocessing.cpu_count() - 1, args.n_sightlines)
    sightlines_obj = Sightlines(**vars(args))
    with multiprocessing.Pool(n_cores) as pool:
        sightlines_obj.parallel_raytrace()
    sightlines_obj.apply_calibration()
