def validate(opt, trainer, model, data):
    if not opt.no_validate and not trainer.interrupted:
        trainer.validate(model, data)

def fit(opt, trainer, model, data):
    if opt.train:
        for tl in trainer.loggers:
            unlockFlag = getattr(tl, 'unlockFlag', lambda *args, **kwargs: None)
            unlockFlag()
        trainer.fit(model, data)
        for tl in trainer.loggers:
            lockFlag = getattr(tl, 'lockFlag', lambda *args, **kwargs: None)
            lockFlag()

def test(opt, trainer, model, data):
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)
    

