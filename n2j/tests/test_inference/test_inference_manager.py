"""Tests for the n2j.trainval_data.graphs.cosmodc2_graph.CosmoDC2Graph class

"""

import os
import unittest
import shutil
import numpy as np
import pandas as pd
import scipy.stats
from n2j.inference.inference_manager import InferenceManager
import n2j.data as in_data


class TestInferenceManager(unittest.TestCase):
    """A suite of tests verifying CosmoDC2Graph class methods

    """

    @classmethod
    def setUpClass(cls):
        """InferenceManager object to test

        """
        infer_obj = InferenceManager('cuda',
                                     checkpoint_dir='results/E3',
                                     seed=1028)
        cls.infer_obj = infer_obj

    def test_load_dataset(self):
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
        IN_DIR = in_data.__path__[0]  # where raw data lies
        TRAIN_HP = [10327]
        VAL_HP = [10326]
        N_TRAIN = [20000]
        N_VAL = 1000
        BATCH_SIZE = 1000  # min(N_TRAIN//5, 50)

        SUB_TARGET = ['final_kappa', ]  # 'final_gamma1', 'final_gamma2']
        SUB_TARGET_LOCAL = ['stellar_mass', 'redshift']
        norm_obj = scipy.stats.norm(loc=0.01, scale=0.03)
        # Training
        self.infer_obj.load_dataset(
                                    dict(features=features,
                                         raytracing_out_dirs=[os.path.join(IN_DIR, f'cosmodc2_{hp}/Y_{hp}') for hp in TRAIN_HP],
                                         healpixes=TRAIN_HP,
                                         n_data=N_TRAIN,
                                         aperture_size=1.0,
                                         subsample_pdf_func=norm_obj.pdf,
                                         stop_mean_std_early=False,
                                         in_dir=IN_DIR),
                                    sub_features=sub_features,
                                    sub_target=SUB_TARGET,
                                    sub_target_local=SUB_TARGET_LOCAL,
                                    is_train=True,
                                    batch_size=BATCH_SIZE,
                                    rebin=False,
                                    )
        # Test
        self.infer_obj.load_dataset(
                                    dict(features=features,
                                         raytracing_out_dirs=[os.path.join(IN_DIR, f'cosmodc2_{hp}/Y_{hp}') for hp in VAL_HP],
                                         healpixes=VAL_HP,
                                         n_data=[N_VAL]*len(VAL_HP),
                                         aperture_size=1.0,
                                         in_dir=IN_DIR),
                                    sub_features=sub_features,
                                    sub_target=SUB_TARGET,
                                    sub_target_local=SUB_TARGET_LOCAL,
                                    is_train=False,
                                    batch_size=N_VAL,  # FIXME: must be same as train
                                    )

    def test_configure_model(self):
        pass

    def test_load_checkpoint(self):
        pass

    def test_get_bnn_kappa(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()
