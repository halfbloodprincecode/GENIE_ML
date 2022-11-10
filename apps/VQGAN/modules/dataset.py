from apps.VQGAN.data.utils import custom_collate
from apps.VQGAN.modules.configuration import Config
from utils.ptDataset import DataModuleFromConfigBase, WrappedDatasetBase

class WrappedDataset(WrappedDatasetBase):
    pass

class DataModuleFromConfig(DataModuleFromConfigBase):
    def __init__(self, **kwargs):
        kwargs['custom_collate'] = custom_collate
        kwargs['instantiate_from_config'] = Config.instantiate_from_config
        super().__init__(**kwargs)
        self.wrap_cls = WrappedDataset
