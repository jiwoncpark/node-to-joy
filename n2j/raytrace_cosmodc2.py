"""Script to raytrace through cosmoD2 sightlines

Example
-------
To run this script, pass in the destination directory as the argument::

    $ python n2j/raytrace_cosmodc2.py <dest_dir>

"""

import functools
import argparse
from tqdm import tqdm
import multiprocessing
from n2j.trainval_data import get_catalog

class ParallelRaytracer:
    def __init__(self,
                 out_dir: str,
                 n_sightlines: int,
                 fov: float,
                 n_samples: int,
                 map_kappa: bool = False,
                 map_gamma: bool = False,
                 diff: float = 1.0):
        # Wrapper attributes
        self.fov = fov
        self.n_samples = n_samples
        self.n_sightlines = n_sightlines
        self.map_kappa = map_kappa
        self.map_gamma = map_gamma
        self.diff = diff
        # Catalog-specific operations
        self.catalog = get_catalog('cosmodc2')(args.out_dir)
        self.catalog.healpix = 10450
        self.catalog.get_sightlines_on_grid(self.n_sightlines,
                                            self.fov*0.5/60.0)

    def raytrace_all(self):
        single = functools.partial(self.catalog.raytrace_single,
                                   fov=self.fov,
                                   n_samples=self.n_samples,
                                   diff=self.diff,
                                   map_kappa=self.map_kappa,
                                   map_gamma=self.map_gamma,
                                   )
        return list(tqdm(pool.imap(single, range(self.n_sightlines)),
                         total=self.n_sightlines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir',
                        help='destination folder for output data')
    parser.add_argument('--fov', default=6.0, dest='fov',
                        type=float,
                        help='field of view in arcmin (Default: 6.0)')
    parser.add_argument('--map_kappa', default=False, dest='map_kappa',
                        type=bool,
                        help='whether to generate grid maps of kappa'
                             ' (Default: False)')
    parser.add_argument('--map_gamma', default=False, dest='map_gamma',
                        type=bool,
                        help='whether to generate grid maps of gamma'
                             ' (Default: False)')
    parser.add_argument('--n_sightlines', default=100, dest='n_sightlines',
                        type=int,
                        help='number of sightlines to raytrace through'
                             ' (Default: 1000)')
    parser.add_argument('--n_samples', default=1000, dest='n_samples',
                        type=int,
                        help='number of kappa samples per sightline'
                             ' (Default: 1000)')
    parser.add_argument('--diff', default=1.0, dest='diff',
                        type=float,
                        help='step size for kappa differential, in asec'
                             ' (Default: 1)')
    args = parser.parse_args()

    profile = False
    if profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

    cosmodc2 = get_catalog('cosmodc2')(args.out_dir)
    cosmodc2.healpix = 10450
    raytracing_kwargs = {k: v for k, v in vars(args).items() if k != 'out_dir'}
    cosmodc2.raytrace_all(**raytracing_kwargs)

    #n_cores = min(multiprocessing.cpu_count() - 1, args.n_sightlines)
    #parallel_raytracer = ParallelRaytracer(**vars(args))
    #with multiprocessing.Pool(n_cores) as pool:
    #    parallel_raytracer.raytrace_all()

    if profile:
        pr.disable()
        pr.print_stats(sort='cumtime')