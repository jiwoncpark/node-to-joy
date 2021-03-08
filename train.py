"""Script to train the Bayesian GNN

"""
from n2j.trainer import Trainer


if __name__ == '__main__':
    features = ['ra_true', 'dec_true']
    features += ['ellipticity_1_true', 'ellipticity_2_true']
    features += ['size_true']
    features += ['mag_{:s}_lsst'.format(b) for b in 'ugrizY']
    trainer = Trainer('cuda', checkpoint_dir='test_run', seed=1234)
    trainer.load_dataset(features,
                         raytracing_out_dir='cosmodc2_raytracing_10450',
                         healpix=10450,
                         n_data=20000,
                         is_train=True,
                         batch_size=100,
                         aperture_size=1.0,
                         stop_mean_std_early=True)
    if True:  # FIXME: must be run after train
        trainer.load_dataset(features,
                             raytracing_out_dir='cosmodc2_raytracing_9559',
                             healpix=9559,
                             n_data=1000,
                             is_train=False,
                             batch_size=100,  # FIXME: must be same as train
                             aperture_size=1.0)
    trainer.configure_loss_fn('DoubleGaussianNLL')
    trainer.configure_model('GATNet', {'hidden_channels': 16,
                                       'n_layers': 3,
                                       'dropout': 0.05,
                                       'kwargs': {'concat': False,
                                                  'heads': 4}})
    trainer.configure_optim({'lr': 1.e-3, 'weight_decay': 1.e-5},
                            {'factor': 0.5, 'min_lr': 1.e-7})
    if False:
        trainer.load_state('/home/jwp/stage/sl/n2j/test_run/DoubleGaussianNLL_epoch=0_03-05-2021_23:41.mdl')
    trainer.train(n_epochs=50)
    print(trainer)
