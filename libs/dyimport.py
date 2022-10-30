import re
import importlib
from libs.basicHR import EHR
from libs.basicIO import pathBIO

def Import(fpath, reload=False, partialFlag=True):
    if partialFlag: # e.g. fpath="articles.taming_transformers.taming.models.vqgan.VQModel"
        _module, _cls = fpath.rsplit('.', 1)
        if reload:
            module_imp = importlib.import_module(_module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(_module, package=None), _cls)
    else:
        return importlib.import_module(fpath, package=None)


# def Import(path, fname=None):
#     path = pathBIO(path)
#     fname = path.split('/')[-1].replace('.py', '')
#     return fname
#     try:
#         spec = importlib.util.spec_from_file_location(fname, path)
#         f = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(f)
#         return f
#     except Exception as e:
#         EHR(e)