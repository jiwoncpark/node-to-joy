"""Script to train the Bayesian GNN

"""
from n2j.trainer import Trainer


if __name__ == '__main__':
    # Features to compile
    features = ['ra_true', 'dec_true']
    features += ['ellipticity_1_true', 'ellipticity_2_true']
    features += ['size_true']
    features += ['mag_{:s}_lsst'.format(b) for b in 'ugrizy']
    # Features to train on
    sub_features = ['ra_true', 'dec_true']
    sub_features += ['size_true']
    sub_features += ['mag_{:s}_lsst'.format(b) for b in 'i']
    trainer = Trainer('cuda', checkpoint_dir='test_run', seed=1234)
    trainer.load_dataset(features,
                         raytracing_out_dir='cosmodc2_raytracing_10450',
                         healpix=10450,
                         n_data=100,
                         is_train=True,
                         batch_size=50,
                         aperture_size=1.0,
                         sub_features=sub_features,
                         stop_mean_std_early=True)
    # FIXME: must be run after train
    trainer.load_dataset(features,
                         raytracing_out_dir='cosmodc2_raytracing_9559',
                         healpix=9559,
                         n_data=100,
                         is_train=False,
                         batch_size=50,  # FIXME: must be same as train
                         aperture_size=1.0,
                         sub_features=sub_features)
    trainer.configure_loss_fn('FullRankGaussianNLL')
    if True:
        trainer.configure_model('GATNet',
                                {'hidden_channels': 128,
                                 'n_layers': 3,
                                 'dropout': 0.0,
                                 'kwargs': {'concat': False, 'heads': 2}})
    trainer.configure_optim({'lr': 1.e-4, 'weight_decay': 1.e-5},
                            {'factor': 0.5, 'min_lr': 1.e-7, 'patience': 10})
    if False:
        trainer.load_state('/home/jwp/stage/sl/n2j/test_run/DoubleGaussianNLL_epoch=0_03-05-2021_23:41.mdl')
    trainer.train(n_epochs=100)
    print(trainer)
