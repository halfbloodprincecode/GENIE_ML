from loguru import logger
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

class GenieLoggerBase(Logger):
    def __init__(
            self, 
            save_dir: str = None,
            name: Optional[str] = 'GeineLogs',
            agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
            agg_default_func: Optional[Callable[[Sequence[float]], float]] = None,
            # **kwargs: Any
        ):
        super().__init__(agg_key_funcs=agg_key_funcs, agg_default_func=agg_default_func)
        self._save_dir = save_dir
        self._name = name or ''

    @property
    def name(self):
        return "GenieLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        logger.warning('log_hyperparams')
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        logger.warning('log_hyperparams | metrics={}'.format(metrics))
        pass

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        super().save()
        logger.warning('save')
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        super().finalize(status)
        logger.warning('finalize | status={}'.format(status))
        pass