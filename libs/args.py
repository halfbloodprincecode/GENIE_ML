from loguru import logger
from pytorch_lightning.trainer import Trainer
from argparse import ArgumentParser, Namespace, ArgumentTypeError

class Args():
    __ns = []
    __allKeys = []
    def __init__(self, argClass):
        """
        """
        super(Args, self).__init__()
        self._argClass = argClass
        sp = '-' if self._argClass else ''
        self._f = lambda key: ('-' if len(key)==1 else '--') + self._argClass + sp + key
        self._args = ArgumentParser()

    def add_argument(self, *keys, **kwargs):
        """ *keys: key0, key1, key2, ..., keyn ==> key1 consider as master key when number of key more than one otherwise key0 consider as master key."""
        keyAlias = [self._f(keyi) for keyi in keys]
        keyAlias = list(map(lambda keyName: keyName.upper(), keyAlias)) + keyAlias
        # logger.info(keyAlias)
        return self._args.add_argument(*keyAlias, **kwargs)

    def parser(self, parse_known_args=True):
        if parse_known_args:
            ns_config =  self._args.parse_known_args()[0]
            Args.__ns.append(ns_config)
            return ns_config
    
    @staticmethod
    def nondefault_trainer_args(opt):
        parser = ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        args = parser.parse_args([])
        return sorted(k for k in vars(args) if getattr(opt, k, None) != getattr(args, k, None))

    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise ArgumentTypeError('Boolean value expected.')
    
    @staticmethod
    def export():
        dic = {}
        for ns_i in Args.__ns:
            dic.update(**vars(ns_i))

        return Namespace(**dic)