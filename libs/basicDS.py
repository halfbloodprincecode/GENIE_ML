from typing import Union
from attrdict import AttrDict
from argparse import Namespace
from functools import partial, wraps
from typing import Any, Callable, Optional, Union

# def ns_add(ns: Namespace, *posargs: Union[Namespace, dict]):
#     pass

def dotdict(d: dict, flag=None):
    flag = bool(True if flag is None else flag)
    if isinstance(d, dict) and flag:
        return AttrDict(d)
    return d

def dict2ns(d: dict):
    if isinstance(d, dict):
        return Namespace(**d)
