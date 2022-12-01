import os
from loguru import logger
from omegaconf import OmegaConf
from utils.ptCallback import SetupCallbackBase, CustomProgressBarBase, ImageLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor


class SetupCallback(SetupCallbackBase):
    def on_fit_start(self, trainer, pl_module):
        logger.warning('-> ***on_fit_start***')
        super().on_fit_start(trainer, pl_module)

class CustomProgressBar(CustomProgressBarBase):
    pass

class ImageLogger(ImageLoggerBase):
    pass