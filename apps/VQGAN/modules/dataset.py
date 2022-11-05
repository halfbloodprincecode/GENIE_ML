from utils.ptDataset import DataModuleFromConfigBase, WrappedDatasetBase

class WrappedDataset(WrappedDatasetBase):
    pass

class DataModuleFromConfig(DataModuleFromConfigBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wrap_cls = WrappedDataset
