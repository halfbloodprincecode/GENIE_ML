from typing import Union
from attrdict import AttrDict
from argparse import Namespace

# def ns_add(ns: Namespace, *posargs: Union[Namespace, dict]):
#     pass

def dotdict(d: dict):
    if isinstance(d, dict):
        return AttrDict(d)
    return d

def dict2ns(d: dict):
    if isinstance(d, dict):
        return Namespace(**d)