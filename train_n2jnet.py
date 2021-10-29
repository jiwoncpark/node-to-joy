"""Script to train the Bayesian GNN

"""
import os
from scipy import stats
from n2j.trainer import Trainer
from n2j.config_utils import get_config
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    cfg = get_config()
    trainer = Trainer(**cfg['trainer'])
    norm_obj = getattr(stats, cfg['data']['train_dist_name'])(**cfg['data']['train_dist_kwargs'])
    train_raytracing = [os.path.join(cfg['data']['in_dir'],
                                     f'cosmodc2_{hp}', 
                                     f'Y_{hp}') for hp in cfg['data']['train_hp']]
    trainer.load_dataset(dict(features=cfg['data']['features'],
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
                         noise_kwargs=cfg['data']['noise_kwargs'],
                         detection_kwargs=cfg['data'].get('detection_kwargs', {}),
                         )
    # FIXME: must be run after train
    val_raytracing = [os.path.join(cfg['data']['in_dir'],
                                   f'cosmodc2_{hp}/Y_{hp}') for hp in cfg['data']['val_hp']]
    trainer.load_dataset(dict(features=cfg['data']['features'],
                              raytracing_out_dirs=val_raytracing,
                              healpixes=cfg['data']['val_hp'],
                              n_data=cfg['data']['n_val'],
                              aperture_size=1.0,
                              subsample_pdf_func=norm_obj.pdf,
                              n_subsample=cfg['data']['n_subsample_val'],
                              in_dir=cfg['data']['in_dir']),
                         sub_features=cfg['data']['sub_features'],
                         sub_target=cfg['data']['sub_target'],
                         sub_target_local=cfg['data']['sub_target_local'],
                         is_train=False,
                         batch_size=cfg['data']['val_batch_size'],
                         rebin=False,
                         noise_kwargs=cfg['data']['noise_kwargs'],
                         detection_kwargs=cfg['data'].get('detection_kwargs', {}),
                         )
    if False:
        print(trainer.train_dataset[0].y_local)
        for b in trainer.train_loader:
            print(b.x.shape, b.y_local.shape, b.y.shape, b.batch.shape)
            print(b.y_local[:5, 0])
            break

    model_kwargs = dict(dim_in=trainer.X_dim,
                        dim_out_local=len(cfg['data']['sub_target_local']),
                        dim_out_global=len(cfg['data']['sub_target']),
                        **cfg['model']
                        )
    trainer.configure_model('N2JNet', model_kwargs)

    trainer.configure_optim(**cfg['optimization'])

    if cfg['resume_from']['checkpoint_path']:
        trainer.load_state(cfg['resume_from']['checkpoint_path'])

    def get_lr(gamma, optimizer):
        return [group['lr'] * gamma for group in optimizer.param_groups]
    #for param_group, lr in zip(trainer.optimizer.param_groups,
    #                           get_lr(0.5, trainer.optimizer)):
    #    param_group['lr'] = lr
    trainer.train(n_epochs=cfg['n_epochs'])
