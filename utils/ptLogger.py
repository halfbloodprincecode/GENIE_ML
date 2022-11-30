from loguru import logger
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

class GenieLoggerBase(Logger):
    def __init__(
        self, 
        save_dir: str = None,
        name: Optional[str] = 'GeineLogger',
        fn_name: Optional[str] = '_genie',
        version: str = '0.1',
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._save_dir = save_dir
        self._name = name
        self._fn_name = fn_name
        self._version = version

    @property
    def name(self):
        return self._name
    
    @property
    def fn_name(self):
        return self._fn_name

    @property
    def version(self):
        # Return the experiment version, int or str.
        return self._version

    @property
    def save_dir(self) -> Optional[str]:
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data locally."""
        return self._save_dir

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        logger.critical('log_hyperparams | params={}'.format(params))
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        logger.critical('log_hyperparams | metrics={}'.format(metrics))
        pass

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        super().save()
        logger.critical('save')
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        super().finalize(status)
        logger.critical('finalize | status={}'.format(status))
        pass