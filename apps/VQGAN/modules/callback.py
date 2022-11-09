import os
from omegaconf import OmegaConf
from utils.ptCallback import SetupCallbackBase, ImageLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor


class SetupCallback(SetupCallbackBase):
    def on_pretrain_routine_start(self, trainer, pl_module):
        # super().on_pretrain_routine_start(trainer, pl_module)
        print('-> ***on_pretrain_routine_start***')

class ImageLogger(ImageLoggerBase):
    pass