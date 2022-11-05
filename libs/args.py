from argparse import ArgumentParser
from abc import ABC, abstractmethod
from pytorch_lightning.trainer import Trainer

class ParserBasic(ABC):
    def __new__(cls, **kwargs):
        super().__new__(cls)
        return cls.parser(**kwargs)
    
    @classmethod
    def nondefault_trainer_args(cls, opt):
        parser = ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        args = parser.parse_args([]) # this is specefic syntax and its mean only return know params[Trainer params] with default values.
        return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

    @classmethod
    def parser(cls, **kwargs):
        ctlFlag = kwargs.get('ctlFlag', True)
        parser = cls.get_parser(**kwargs)
        parser = Trainer.add_argparse_args(parser)
        opt, unknown = parser.parse_known_args()
        
        if ctlFlag:
            return opt, unknown, cls.ctl_parser(opt, unknown, **kwargs)
        return opt, unknown
    
    @classmethod
    @abstractmethod
    def get_parser(cls, **kwargs):
        pass
    
    @classmethod
    @abstractmethod
    def ctl_parser(cls, opt, unknown, **kwargs):
        pass