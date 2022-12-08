import argparse
from os import environ
from os.path import join
from loguru import logger
from omegaconf import OmegaConf
from libs.dyimport import Import
from libs.args import ParserBasic
from pytorch_lightning.trainer import Trainer

class ConfigBase:
    def __new__(cls, **kwargs):
        super().__new__(cls)
        return cls.fn(**kwargs)
    
    @classmethod
    def instantiate_from_config(cls, config):
        if not 'target' in config:
            raise KeyError('Expected key `target` to instantiate.')
        return Import(config['target'])(**config.get('params', dict()))

    @classmethod
    def fn(cls, **kwargs):
        now = kwargs['now']
        opt = kwargs['opt']
        unknown = kwargs['unknown']
        cfgdir = kwargs['cfgdir']
        logdir = kwargs['logdir']
        ckptdir = kwargs['ckptdir']
        nowname = kwargs['nowname']
        nondefault_trainer_args = kwargs['nondefault_trainer_args']

        # configure model&trainer ##########################################
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop('lightning', OmegaConf.create())
        
        # print(config)
        # print('='*60)
        # print(lightning_config)
        # print('='*60)
        # input()
        # return
        
        # merge trainer cli with config
        trainer_config = lightning_config.get('trainer', OmegaConf.create())
        
        # correction phase on trainer_config
        trainer_config['distributed_backend'] = 'ddp' # default to ddp
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not 'gpus' in trainer_config:
            del trainer_config['distributed_backend']
            cpu = True
        else:
            gpuinfo = trainer_config['gpus']
            logger.info(f'Running on GPUs {gpuinfo}')
            cpu = False
        
        logger.warning('trainer_config={}'.format(trainer_config))
        
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = cls.instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # *****************[default logger]*****************
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            'wandb': {
                'target': 'pytorch_lightning.loggers.WandbLogger',
                'params': {
                    'name': nowname,
                    'save_dir': logdir,
                    'offline': opt.debug,
                    'id': nowname,
                }
            },
            'testtube': {
                'target': 'pytorch_lightning.loggers.TestTubeLogger',
                'params': {
                    'name': 'testtube',
                    'save_dir': logdir,
                }
            },
            'tensorboard': {
                'target': 'pytorch_lightning.loggers.TensorBoardLogger',
                'params': {
                    'name': nowname,
                    'save_dir': logdir,
                }
            },
            'genie': {
                'target': 'apps.VQGAN.modules.genie_logger.GenieLogger',
                'params': {
                    'name': nowname,
                    'save_dir': logdir,
                }
            },
        }
         
        default_logger_cfg = default_logger_cfgs[opt.logger_ml] # default is: 'genie'
        logger_cfg = lightning_config.get('logger', OmegaConf.create()) # lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs['logger'] = cls.instantiate_from_config(logger_cfg)
        

        # *****************[model checkpoint]*****************
        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            'target': 'apps.VQGAN.modules.callback.ModelCheckpoint',
            'params': {
                'monitor': 'val/total_loss_epoch', #val/total_loss_epoch # this line maybe had changed!!
                'mode': 'min',
                'lastname': 'best',
                'every_n_epochs': 0,
                'dirpath': ckptdir,
                'verbose': True,
                'save_last': True
            }
        }
        if hasattr(model, 'monitor'):
            logger.critical(f'||||Monitoring {model.monitor} as checkpoint metric.')
            default_modelckpt_cfg['params']['monitor'] = model.monitor
            default_modelckpt_cfg['params']['save_top_k'] = 3

        modelckpt_cfg = lightning_config.get('modelcheckpoint', OmegaConf.create()) # lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        _checkpoint_callback = cls.instantiate_from_config(modelckpt_cfg)
        # trainer_kwargs['checkpoint_callback'] = cls.instantiate_from_config(modelckpt_cfg)
        logger.warning('lightning_config.modelcheckpoint={}'.format( lightning_config.get('modelcheckpoint',{}) ))
        logger.warning('modelckpt_cfg={}'.format(modelckpt_cfg))

        # *****************[sets up log directory]*****************
        # add callback which sets up log directory
        default_callbacks_cfg = {
            'setup_callback': {
                'target': 'apps.VQGAN.modules.callback.SetupCallback',
                'params': {
                    'resume': opt.resume,
                    'now': now,
                    'logdir': logdir,
                    'ckptdir': ckptdir,
                    'cfgdir': cfgdir,
                    'config': config,
                    'lightning_config': lightning_config,
                }
            },
            'custom_progressBar': {
                'target': 'apps.VQGAN.modules.callback.CustomProgressBar',
                'params': {}
            },
            # 'image_logger': {
            #     'target': 'apps.VQGAN.modules.callback.ImageLogger',
            #     'params': {
            #         'batch_frequency': 500,# 750,
            #         'max_images': 4,
            #         'clamp': True
            #     }
            # },
            # 'learning_rate_logger': { # it must be uncomment!!!!!!
            #     'target': 'apps.VQGAN.modules.callback.LearningRateMonitor',
            #     'params': {
            #         'logging_interval': 'step',
            #         #'log_momentum': True
            #     }
            # },
            'cb': {
                'target': 'apps.VQGAN.modules.callback.CB',
                'params': {}
            },
            'UnconditionalCheckpointing': {
                'target': 'apps.VQGAN.modules.callback.ModelCheckpoint',
                'params': {
                    'lastname': 'last',
                    'every_n_epochs': 0,
                    'dirpath': ckptdir,
                    'save_last': True,
                    'verbose': True
                }
            },
        }
        callbacks_cfg = lightning_config.get('callbacks', OmegaConf.create()) # lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs['callbacks'] = [cls.instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        trainer_kwargs['callbacks'].append(_checkpoint_callback)

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        for tl in trainer.loggers:
            handiCall = getattr(tl, 'setter_handiCall', lambda *args, **kwargs: None)
            handiCall(_trainer_obj=trainer)

        # configure data #############################################
        data = cls.instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        # data.prepare_data()
        data.setup()

        # configure learning rate ####################################
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(',').split(','))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.get('accumulate_grad_batches', 1)
        logger.info(f'accumulate_grad_batches = {accumulate_grad_batches}')
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        logger.info('Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)'.format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        
        return model, trainer, data