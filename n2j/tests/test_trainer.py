"""Tests for the n2j.trainval_data.graphs.cosmodc2_graph.CosmoDC2Graph class

"""

import os
import unittest
import shutil
import numpy as np
import pandas as pd
import scipy.stats
from n2j.trainer import Trainer
import n2j.data as in_data
from n2j.trainval_data.utils.running_stats import RunningStats


class TestTrainer(unittest.TestCase):
    """A suite of tests verifying CosmoDC2Graph class methods

    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests
        """
        cls.checkpoint_dir = os.path.join('test_trainer_dir')
        cls.IN_DIR = in_data.__path__[0]  # where raw data lies
        cls.TRAIN_HP = [10327]
        cls.VAL_HP = [10450]
        cls.N_TRAIN = [50000]
        cls.N_VAL = 1000
        cls.SUB_TARGET = ['final_kappa', ]  # 'final_gamma1', 'final_gamma2']
        cls.SUB_TARGET_LOCAL = ['stellar_mass', 'redshift']

    def setUp(self):
        self.trainer = Trainer('cuda',
                               checkpoint_dir=self.checkpoint_dir,
                               seed=1028)

    def test_load_dataset_subsampling(self):
        """Test if the subsampled data follows the specified distribution

        """
        features = ['galaxy_id', 'ra', 'dec', 'redshift']
        features += ['ra_true', 'dec_true', 'redshift_true']
        features += ['ellipticity_1_true', 'ellipticity_2_true']
        features += ['bulge_to_total_ratio_i']
        features += ['ellipticity_1_bulge_true', 'ellipticity_1_disk_true']
        features += ['ellipticity_2_bulge_true', 'ellipticity_2_disk_true']
        features += ['shear1', 'shear2', 'convergence']
        features += ['size_bulge_true', 'size_disk_true', 'size_true']
        features += ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']
        # Features to train on
        sub_features = ['ra_true', 'dec_true']
        # sub_features += ['size_true']
        # sub_features += ['ellipticity_1_true', 'ellipticity_2_true']
        sub_features += ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']
        ray_dirs = [os.path.join(self.IN_DIR, f'cosmodc2_{hp}/Y_{hp}') for hp in self.TRAIN_HP]
        # User-specified distribution
        norm_obj = scipy.stats.norm(loc=0.01, scale=0.03)
        self.trainer.load_dataset(
                                  dict(features=features,
                                       raytracing_out_dirs=ray_dirs,
                                       healpixes=self.TRAIN_HP,
                                       n_data=self.N_TRAIN,
                                       aperture_size=1.0,
                                       subsample_pdf_func=norm_obj.pdf,
                                       stop_mean_std_early=False,
                                       in_dir=self.IN_DIR),
                                  sub_features=sub_features,
                                  sub_target=self.SUB_TARGET,
                                  sub_target_local=self.SUB_TARGET_LOCAL,
                                  is_train=True,
                                  batch_size=self.BATCH_SIZE,
                                  rebin=False,
                                  )
        rs = RunningStats(loader_dict=dict(Y=lambda b: b.y))
        for i, b in enumerate(self.trainer.train_loader):
            rs.update(b, i)
        np.testing.assert_array_almost_equal(rs.stats['Y_mean'], 0.01, decimal=2)
        np.testing.assert_array_almost_equal(rs.stats['Y_var']**0.5, 0.03, decimal=2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.raytracing_out_dir)


if __name__ == '__main__':
    unittest.main()
