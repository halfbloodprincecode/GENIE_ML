from loguru import logger
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class GenieLoggerBase(Logger):
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
        logger.warning('save')
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        logger.warning('finalize | status={}'.format(status))
        pass