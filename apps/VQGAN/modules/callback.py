import os
from omegaconf import OmegaConf
from utils.ptCallback import SetupCallbackBase
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor


class SetupCallback(SetupCallbackBase):
    def on_pretrain_routine_start(self, trainer, pl_module):
        # super().on_pretrain_routine_start(trainer, pl_module)
        print('-> ***on_pretrain_routine_start***')