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
    def predefined_args(cls, parser):
        parser.add_argument(
            '-L',
            '--logger_ml',
            type=str,
            default='genie', #'tensorboard',
            help='default_logger_cfgs key name',
        )
        parser.add_argument(
            '-M',
            '--metrics_tbl',
            type=str,
            const=True,
            default=None,
            nargs='?',
            help='metrics table name',
        )
        parser.add_argument(
            '-C',
            '--ckpt-fname',
            type=str,
            default='last',
            help='ckpt fname',
        )
        parser.add_argument(
            '-H',
            '--hash-ignore',
            nargs='*',
            help='hash ignore list for plLogger(Geine)',
            default=list(),
        )
        return parser

    @classmethod
    def parser(cls, **kwargs):
        ctlFlag = kwargs.get('ctlFlag', True)
        predefFlag = kwargs.get('predefFlag', True)
        parser = cls.get_parser(**kwargs)
        if predefFlag:
            parser = cls.predefined_args(parser)
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