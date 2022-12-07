from loguru import logger
from utils.ptCallback import ModelCheckpointBase, CheckpointHandlerBase, SetupCallbackBase, CustomProgressBarBase, ImageLoggerBase, CBBase
# from pytorch_lightning.callbacks import ModelCheckpoint as ModelCheckpointBase, Callback, LearningRateMonitor


class ModelCheckpoint(ModelCheckpointBase):
    pass

class CheckpointHandler(CheckpointHandlerBase):
    pass

class SetupCallback(SetupCallbackBase):
    def on_fit_start(self, trainer, pl_module):
        logger.warning('[child class] -> ***on_fit_start***')
        super().on_fit_start(trainer, pl_module)

class CustomProgressBar(CustomProgressBarBase):
    pass

class ImageLogger(ImageLoggerBase):
    pass

class CB(CBBase):
    pass