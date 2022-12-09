from loguru import logger
from utils.ptCallback import ModelCheckpointBase, SetupCallbackBase, CustomProgressBarBase, ImageLoggerBase, CBBase
# from pytorch_lightning.callbacks import ModelCheckpoint as ModelCheckpointBase, Callback, LearningRateMonitor


class ModelCheckpoint(ModelCheckpointBase):
    pass

class SetupCallback(SetupCallbackBase):
    pass

class CustomProgressBar(CustomProgressBarBase):
    pass

class ImageLogger(ImageLoggerBase):
    pass

class CB(CBBase):
    pass