"""Tests for the n2j.train_utils utility functions

"""

import os
import unittest
import numpy.testing as npt
from n2j import train_utils


class TestTrainUtils(unittest.TestCase):
    """A suite of tests verifying train_utils utility functions

    """

    @classmethod
    def setUpClass(cls):
        """Set global defaults for tests
        """
        from n2j import tests
        cls.cfg_path = os.path.join(tests.__path__[0], 'sample_training_cfg.yml')

    def test_get_train_cfg(self):
        """Test retrieval of various data types from cfg

        """
        cfg = train_utils.get_train_cfg_modular([self.cfg_path])
        npt.assert_equal(cfg['section_0']['str_value'], 'some_parent/some_dir')
        npt.assert_equal(cfg['section_0']['int_list'], [1, 2])
        npt.assert_equal(cfg['section_0']['int_value'], 1000)
        npt.assert_equal(cfg['section_0']['str_list'], ['str_element'])
        npt.assert_equal(cfg['section_0']['multiline_str_list'],
                         ['0', '1', '2', '3', '4', '5', '6', '7'])
        npt.assert_equal(cfg['section_1']['inside_dict'],
                         dict(float_value=0.001,
                              int_value=1,
                              null_value=None))
        npt.assert_equal(cfg['section_1']['inside_dict']['null_value'], None)

    @classmethod
    def tearDownClass(cls):
        pass  # shutil.rmtree(cls.raytracing_out_dir)


if __name__ == '__main__':
    unittest.main()
