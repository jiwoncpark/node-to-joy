import argparse
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('aperture_size', type=float, #nargs='+',
                    help='size of aperture in arcmin')
parser.add_argument('healpix', type=int, #nargs='+',
                    help='healpix ID')
args = parser.parse_args()
#cols = [galaxy_id, position_angle_true, redshift_true, mag_u_lsst, shear_2, ra, shear_2_treecorr, mag_Y_lsst, dec, mag_i_lsst, size_true, mag_g_lsst, mag_r_lsst, redshift, ellipticity_true, convergence, mag_z_lsst, shear_2_phosim, shear_1]
start = time.time()
#print("Filename: los_{0:.4g}arcmin_{:d}.csv".format(args.aperture_size, args.healpix))

cosmodc2 = pd.read_csv('data/cosmodc2/raw/cosmodc2_trainval_{:d}.csv'.format(args.healpix), index_col=None, nrows=50000)
N, n_features = cosmodc2.shape
print("Read in cosmodc2 with {:d} objects".format(N))

cosmodc2[['ra', 'dec']] *= 60 # convert to arcmin
print(cosmodc2[['ra', 'dec']].describe())

los_df = pd.DataFrame(np.nan, index=np.arange(N), columns=['N1', 'N2', 'zeta1', 'zeta2'])

# Boostrap-resample from their R-band magnitude errors, get B=5 realizations of R
B = 1 # TODO: modify implementation for B>1
cosmodc2['mag_r_err'] = 0.0 #0.005 # artificial
for b in range(B):
    mag_r = cosmodc2['mag_r_lsst'].values 
    mag_r += np.random.randn(*mag_r.shape)*cosmodc2['mag_r_err']

    kd_tree = KDTree(cosmodc2[['ra', 'dec']].values) 

    # Use R-band data to count objects with distance > 3'', < 120'' from the lens with 18 < R < 23
    # self is included
    
    for i in tqdm(range(N)):
        this_pos = cosmodc2.loc[i, ['ra', 'dec']].values.reshape([1, -1]) # [1, 2]
        distances_to_this_lens = np.linalg.norm(cosmodc2[['ra', 'dec']].values - this_pos, axis=1) # [N,]
        keep_dist = np.logical_and(distances_to_this_lens < args.aperture_size, distances_to_this_lens > 0.05)
        keep_r = np.logical_and(mag_r < 23.0, mag_r > 18.0)
        keep = np.logical_and(keep_dist, keep_r)
        neighbors = cosmodc2.loc[keep, :] # [n_neighbors,]
        #distances, neighbors = kd_tree.query(smaller_cosmodc2.loc[:, ['ra', 'dec']].values, k=N, distance_upper_bound=2.0, p=2) # [n_lenses, n_neighbors]
        dist = distances_to_this_lens[keep]
        # N1: total number of galaxies
        N1 = neighbors.shape[0]
        los_df.loc[i, 'N1'] = N1
        # N2: inverse projected distance number count
        N2 = np.sum(1.0/dist)
        los_df.loc[i, 'N2'] = N2

    # Get n_calib random objects from the whole catalog

    # zeta1: N1/median N1 in random
    los_df['zeta1'] = los_df['N1']/np.median(los_df['N1'].values) 
    # zeta2: N2/median N2 in random
    los_df['zeta2'] = los_df['N2']/np.median(los_df['N2'].values) 

    los_df.to_csv('los_{0:.4g}arcmin.csv'.format(args.aperture_size), index=None)
end = time.time()
print("Took {:f} min".format((end - start)/60.0)) # min