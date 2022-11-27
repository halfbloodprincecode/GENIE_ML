import os
from loguru import logger
from omegaconf import OmegaConf
from utils.ptCallback import SetupCallbackBase, ImageLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor


class SetupCallback(SetupCallbackBase):
    def on_fit_start(self, trainer, pl_module):
        # super().on_fit_start(trainer, pl_module)
        logger.warning('-> ***on_fit_start***')

class ImageLogger(ImageLoggerBase):
    pass