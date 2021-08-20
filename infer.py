"""Script to perform hierarchical inference with the trained Bayesian GNN

"""
import os
import sys
import numpy as np
from scipy import stats
from n2j.inference.inference_manager import InferenceManager
from n2j.config_utils import get_config

if __name__ == '__main__':
    cfg = get_config()
    infer_obj = InferenceManager(checkpoint_dir=cfg['trainer']['checkpoint_dir'],
                                 **cfg['inference_manager'])
    # Load training stats (for normalizing data)
    norm_obj = getattr(stats, cfg['data']['train_dist_name'])(**cfg['data']['train_dist_kwargs'])
    train_raytracing = [os.path.join(cfg['data']['in_dir'],
                                     f'cosmodc2_{hp}/Y_{hp}') for hp in cfg['data']['train_hp']]
    infer_obj.load_dataset(
                           dict(features=cfg['data']['features'],
                                raytracing_out_dirs=train_raytracing,
                                healpixes=cfg['data']['train_hp'],
                                n_data=cfg['data']['n_train'],
                                aperture_size=1.0,
                                subsample_pdf_func=norm_obj.pdf,
                                n_subsample=cfg['data']['n_subsample_train'],
                                stop_mean_std_early=False,
                                in_dir=cfg['data']['in_dir']),
                           sub_features=cfg['data']['sub_features'],
                           sub_target=cfg['data']['sub_target'],
                           sub_target_local=cfg['data']['sub_target_local'],
                           is_train=True,
                           batch_size=cfg['data']['batch_size'],
                           num_workers=cfg['data']['num_workers'],
                           rebin=False,
                           noise_kwargs=cfg['data']['noise_kwargs']
                           )
    # Load test set
    norm_obj_test = getattr(stats, cfg['test_data']['dist_name'])(**cfg['test_data']['dist_kwargs'])
    test_raytracing = [os.path.join(cfg['data']['in_dir'],
                                    f'cosmodc2_{hp}/Y_{hp}') for hp in cfg['test_data']['test_hp']]
    infer_obj.load_dataset(dict(features=cfg['data']['features'],
                                raytracing_out_dirs=test_raytracing,
                                healpixes=cfg['test_data']['test_hp'],
                                n_data=cfg['test_data']['n_test'],
                                aperture_size=1.0,
                                subsample_pdf_func=norm_obj_test.pdf,
                                n_subsample=cfg['test_data']['n_subsample_test'],
                                in_dir=cfg['data']['in_dir']),
                           sub_features=cfg['data']['sub_features'],
                           sub_target=cfg['data']['sub_target'],
                           sub_target_local=cfg['data']['sub_target_local'],
                           is_train=False,
                           batch_size=cfg['test_data']['batch_size'],
                           noise_kwargs=cfg['data']['noise_kwargs']
                           )
    # Define model
    model_kwargs = dict(
                        dim_in=len(cfg['data']['sub_features']),
                        dim_out_local=len(cfg['data']['sub_target_local']),
                        dim_out_global=len(cfg['data']['sub_target']),
                        **cfg['model']
                        )
    infer_obj.configure_model('N2JNet', model_kwargs)
    # Load trained model
    infer_obj.load_state(cfg['checkpoint_path'])
    # Get summary stats baseline
    infer_obj.get_summary_stats(cfg['summary_stats']['thresholds'])

    # Hierarchical reweighting
    p0 = np.array([[0.01, np.log(0.04)]])
    p0 = p0 + np.random.randn(cfg['extra_mcmc_kwargs']['n_walkers'],
                              2)*np.array([[0.01, 0.5]])
    mcmc_kwargs = dict(p0=p0,
                       chain_path=os.path.join(infer_obj.out_dir, 'omega_chain.h5'),
                       **cfg['extra_mcmc_kwargs']
                       )
    if cfg['run_mcmc']:
        infer_obj.run_mcmc_for_omega_post(n_samples=1000,
                                          n_mc_dropout=20,
                                          mcmc_kwargs=mcmc_kwargs,
                                          interim_pdf_func=norm_obj.pdf,
                                          bounds_lower=np.array([-0.5, -6]),
                                          bounds_upper=np.array([1.5, 0])
                                          )
