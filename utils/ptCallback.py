import os
from os import makedirs, rename
from os.path import join, exists
import torch
import torchvision
import numpy as np
from PIL import Image
from loguru import logger
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only 
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor

class SetupCallbackBase(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            logger.warning('on_fit_start > gRank 0')
            # Create logdirs and save configs
            makedirs(self.logdir, exist_ok=True)
            makedirs(self.ckptdir, exist_ok=True)
            makedirs(self.cfgdir, exist_ok=True)

            logger.info('Project config: {}'.format(self.config))
            OmegaConf.save(self.config, join(self.cfgdir, '{}-project.yaml'.format(self.now)))

            logger.info('Lightning config: {}'.format(self.lightning_config))
            OmegaConf.save(OmegaConf.create({'lightning': self.lightning_config}), join(self.cfgdir, '{}-lightning.yaml'.format(self.now)))
        else:
            logger.warning('vvvvvv on_fit_start > gRank {}'.format(trainer.global_rank))
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and exists(self.logdir):
                logger.warning('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
                dst, name = os.path.split(self.logdir)
                dst = join(dst, 'child_runs', name)
                makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

# TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class ImageLoggerBase(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            # pl.loggers.WandbLogger: self._wandb,
            # pl.loggers.TestTubeLogger: self._testtube,
            # pl.loggers.TensorBoardLogger: self._tb,
            'apps.VQGAN.modules.genie_logger.GenieLogger': self._genie
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError('No way wandb')
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f'{split}/{k}'] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f'{split}/{k}'
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def _tb(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f'{split}/{k}'
            # pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)
    
    @rank_zero_only
    def _genie(self, pl_module, images, batch_idx, split):
        logger.warning('ImageLoggerBase | _genie', type(images), batch_idx)
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f'{split}/{k}'
            # pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        logger.warning('ImageLoggerBase | log_local | save_dir={} | split={}'.format(save_dir, split))
        root = join(save_dir, 'images', split)
        print('++++++++++++++++++++++++++++++')
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = '{}_gs-{:06}_e-{:06}_b-{:06}.png'.format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split='train'):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, 'log_images') and
                callable(pl_module.log_images) and
                self.max_images > 0):
            # _logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            
            _logger = getattr(pl_module.logger, 'fn_name', 'hooooooo')
            logger.error('@@@@@@@@@@ = {} = {}'.format(_logger, getattr(self, _logger)))
            logger_log_images = self[_logger] #self.logger_log_images[_logger] #.get(, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): # ,dataloader_idx
        logger.warning('ImageLoggerBase | on_train_batch_end')
        self.log_img(pl_module, batch, batch_idx, split='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx): # in this case outputs is same as pl_module!!
        logger.warning('ImageLoggerBase | on_validation_batch_end')
        self.log_img(pl_module, batch, batch_idx, split='val')

