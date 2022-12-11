import os
from loguru import logger
from libs.basicHR import EHR 
from libs.basicTime import getTimeHR_V0
from apps.VQGAN.modules.args import Parser
from utils.pl.plApp import validate, fit, test
from apps.VQGAN.modules.configuration import Config
from apps.VQGAN.modules.handler import SignalHandler

class App:
    def __new__(cls, **kwargs):
        super().__new__(cls)
        return cls.main(**kwargs)

    @classmethod
    def main(cls):
        now = getTimeHR_V0()
        opt, unknown, (ckptdir, cfgdir, logdir, nowname) = Parser(now=now)
        
        try:
            model, trainer, data = Config(
                ckptdir=ckptdir, cfgdir=cfgdir, logdir=logdir,
                opt=opt, unknown=unknown, nowname=nowname, now=now,
                nondefault_trainer_args=Parser.nondefault_trainer_args
            )

            SignalHandler(trainer, ckptdir)

            # run
            validate(opt, trainer, model, data)
            try:
                fit(opt, trainer, model, data)
            except Exception:
                SignalHandler.melk()
                raise
            validate(opt, trainer, model, data)
            test(opt, trainer, model, data)
        except Exception as e:
            if opt.debug and trainer.global_rank==0:
                EHR(e)
            raise
        finally:
            # move newly created debug project to debug_runs
            if opt.debug and not opt.resume and trainer.global_rank==0:
                dst, name = os.path.split(logdir)
                dst = os.path.join(dst, 'debug_runs', name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                os.rename(logdir, dst)
    
    @classmethod
    def plot(cls):
        from utils.plots.neonGlowing import Neon
        neon = Neon(xlabel='epoch', ylabel='validation loss')
        neon.plot_metrics(
            # hash = '5ee327ab28725a85bb9fcf6bd3a379052b659d9b',
            # col_names = 'val__aeloss_step, step, epoch',
            hash = '7cea1c511ce7e9bab00be269201cc16effa8ad12',
            col_names = 'val__aeloss_epoch, step, epoch',
            db = '/media/alihejrati/3E3009073008C83B/Code/Genie-ML/logs/vqgan/11/metrics.db',
            smoothing=True,
            # smooth_both=True,
            # label='loss',
        )