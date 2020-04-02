import os, sys
import importlib
import warnings
import json
import glob
import numpy as np
import pandas as pd
from addict import Dict

__all__ = ['TrainValConfig']

class TrainValConfig:
    """Nested dictionary representing the configuration for training, inference, visualization, and analysis

    """
    def __init__(self, user_cfg):
        """
        Parameters
        ----------
        user_cfg : dict or Dict
            user-defined configuration
        
        """
        self.__dict__ = Dict(user_cfg)
        self.validate_user_definition()
        self.preset_default()
        # Data
        self.set_XY_metadata()        
        self.set_model_metadata()

    @classmethod
    def from_file(cls, user_cfg_path):
        """Alternative constructor that accepts the path to the user-defined configuration python file

        Parameters
        ----------
        user_cfg_path : str or os.path object
            path to the user-defined configuration python file

        """
        dirname, filename = os.path.split(os.path.abspath(user_cfg_path))
        module_name, ext = os.path.splitext(filename)
        sys.path.append(dirname)
        if ext == '.py':
            user_cfg_script = importlib.import_module(module_name)
            user_cfg = getattr(user_cfg_script, 'cfg')
            return cls(user_cfg)
        elif ext == '.json':
            with open(user_cfg_path, 'r') as f:
                user_cfg_str = f.read()
            user_cfg = Dict(json.loads(user_cfg_str))
            return cls(user_cfg)
        else:
            raise NotImplementedError("This extension is not supported.")

    def validate_user_definition(self):
        """Check to see if the user-defined config is valid

        """
        pass
        #from .. import losses
        #if not hasattr(losses, self.model.likelihood_class):
        #    raise TypeError("Likelihood class supplied in cfg doesn't exist.")

    def preset_default(self):
        """Preset default config values

        """
        if 'x_path' not in self.data:
            raise ValueError("Must provide training data directory.")
        if 'y_path' not in self.data:
            raise ValueError("Must provide validation data directory.")

    def set_XY_metadata(self):
        """Set general metadata relevant to network architecture and optimization

        """
        # Y metadata
        self.data.Y_dim = len(self.data.Y_cols)
        # Get training-set mean and std for whitening
        train_metadata_path = os.path.join(self.data.train_dir, 'metadata.csv')
        # Data to plot during monitoring
        if self.monitoring.n_plotting > 100:
            warnings.warn("Only plotting allowed max of 100 datapoints during training")
            self.monitoring.n_plotting = 100
        #if self.monitoring.n_plotting > self.optim.batch_size:
        #    raise ValueError("monitoring.n_plotting must be smaller than optim.batch_size")

    def set_model_metadata(self):
        """Set metadata about the network architecture and the loss function (posterior type)

        """
        pass

    def check_train_val_diff(self):
        """Check that the training and validation datasets are different

        """
        if self.data.train_dir == self.data.val_dir:
            warnings.warn("You're training and validating on the same dataset.", UserWarning, stacklevel=2)
