from os import getenv, makedirs
from os.path import join
from loguru import logger
from utils.metrics import Metrics
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

class GenieLoggerBase(Logger):
    def __init__(
        self, 
        save_dir: str = None,
        name: Optional[str] = 'GeineLogger',
        fn_name: Optional[str] = '_genie',
        select_storage: Optional[str] = 'GENIE_ML_STORAGE0',
        version: str = '0.1',
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._save_dir = save_dir
        self._name = name
        self._fn_name = fn_name
        self._version = version
        self.select_storage = select_storage
        self.metrics = None

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
    
    def set_metrics(self, metrics_items):
        db_path_dir = join(getenv(self.select_storage), getenv('GENIE_ML_APP'))
        makedirs(db_path_dir, exist_ok=True)
        self.metrics = Metrics(
            db_path_dir,
            'metrics',
            self._name, # this is `nowname`. (one dir after `logs` in `logdir`) Notic: `nowname` is constant when resuming.
            metrics_items
        )

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        logger.critical('log_hyperparams | params={}'.format(params))

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        metrics_keys = ' | '.join(list(metrics.keys()))
        if self.metrics is None:
            logger.info('metrics.keys={}'.format(metrics_keys))
            # self.set_metrics()
        logger.critical('log_metrics | step={} | metrics={}'.format(step, metrics))
        try:
            logger.debug(self.hparams)
        except Exception as e:
            logger.debug(e)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # super().save()
        # logger.critical('save')
        # try:
        #     print(self.hparams)
        # except Exception as e:
        #     print(e)
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        # logger.critical('finalize | status={}'.format(status))
        # print(self.get('hparams', None))
        # if self._experiment is not None:
        #     self.experiment.flush()
        #     self.experiment.close()
        # self.save()
        pass