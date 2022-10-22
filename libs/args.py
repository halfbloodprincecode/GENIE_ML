from argparse import ArgumentParser, Namespace

class Args():
    __ns = []
    def __init__(self, argClass, strmap=str.upper):
        """
        *use strmap like below if you want:
            strmap=lambda s: s
        """
        super(Args, self).__init__()
        self._argClass = argClass
        sp = '-' if self._argClass else ''
        self._f = lambda key: strmap('--' + self._argClass + sp + key)
        self._args = ArgumentParser()

    def add_argument(self, key: str, **kwargs):
        return self._args.add_argument(self._f(key), **kwargs)
    
    def parser(self, parse_known_args=True):
        if parse_known_args:
            ns_config =  self._args.parse_known_args()[0]
            Args.__ns.append(ns_config)
            return ns_config
    
    @staticmethod
    def export():
        dic = {}
        for ns_i in Args.__ns:
            dic.update(**vars(ns_i))

        return Namespace(**dic)