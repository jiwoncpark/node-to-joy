"""Script to train the Bayesian GNN

"""
import os
import sys
import cProfile
import numpy as np
from scipy import stats
from n2j.trainer import Trainer

if __name__ == '__main__':
    IN_DIR = '/home/jwp/stage/sl/n2j/n2j/data'  # where raw data lies
    TRAIN_HP = [10327]
    VAL_HP = [10450]
    N_TRAIN = [20000]
    N_VAL = 1000
    BATCH_SIZE = 1000  # min(N_TRAIN//5, 50)
    CHECKPOINT_PATH = None  #"/home/jwp/stage/sl/n2j/results/E2/N2JNet_epoch=83_07-08-2021_21:00.mdl"
    SUB_TARGET = ['final_kappa', ]  # 'final_gamma1', 'final_gamma2']
    SUB_TARGET_LOCAL = ['stellar_mass', 'redshift']
    CHECKPOINT_DIR = 'results/E3'
    SKIP_RAYTRACING = True

    ##############
    # Graphs (X) #
    ##############

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
    trainer = Trainer('cuda', checkpoint_dir=CHECKPOINT_DIR, seed=1028)
    norm_obj = stats.norm(loc=0.01, scale=0.03)
    trainer.load_dataset(dict(features=features,
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
    # FIXME: must be run after train
    trainer.load_dataset(dict(features=features,
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
    if False:
        print(trainer.train_dataset[0].y_local)
        for b in trainer.train_loader:
            print(b.x.shape, b.y_local.shape, b.y.shape, b.batch.shape)
            print(b.y_local[:5, 0])
            break

    model_kwargs = dict(dim_in=trainer.X_dim,
                        dim_out_local=len(SUB_TARGET_LOCAL),
                        dim_out_global=len(SUB_TARGET),
                        dim_local=50,
                        dim_global=50,
                        dim_hidden=50,
                        dim_pre_aggr=50,
                        n_iter=5,
                        n_out_layers=5,
                        dropout=0.04,
                        global_flow=False,
                        device_type=trainer.device_type
                        )
    trainer.configure_model('N2JNet', model_kwargs)

    trainer.configure_optim(early_stop_memory=50,
                            weight_local_loss=1.0,
                            optim_kwargs={'lr': 1e-3, 'weight_decay': 1.e-4},
                            lr_scheduler_kwargs={'factor': 0.5, 'min_lr': 1.e-7,
                                                 'patience': 5, 'verbose': True})
    if CHECKPOINT_PATH:
        trainer.load_state(CHECKPOINT_PATH)

    def get_lr(gamma, optimizer):
        return [group['lr'] * gamma for group in optimizer.param_groups]
    #for param_group, lr in zip(trainer.optimizer.param_groups,
    #                           get_lr(0.2, trainer.optimizer)):
    #    param_group['lr'] = lr
    print(len(trainer.train_dataset))
    trainer.train(n_epochs=200, eval_every=2)
    sys.exit()
    print(trainer)
    # Save final validation metrics
    summary = trainer.eval_posterior(epoch_i=trainer.epoch,
                                     n_samples=200,
                                     n_mc_dropout=20,
                                     on_train=False)
    np.save(os.path.join(trainer.checkpoint_dir, 'summary.npy'), summary)
