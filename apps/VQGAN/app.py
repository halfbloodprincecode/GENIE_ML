import os
from libs.basicHR import EHR 
from libs.basicTime import getTimeHR_V0
from apps.VQGAN.modules.args import Parser
from apps.VQGAN.modules.configuration import Config
from apps.VQGAN.modules.handler import SignalHandler

def main():
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
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                SignalHandler.melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            print('trainer.test is called')
            trainer.test(model, data)
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
    
if __name__ == '__main__':
    main()