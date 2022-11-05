import re
import importlib
from libs.basicHR import EHR
from libs.basicIO import pathBIO

def __imp(adr_path, package, embedParams):
    embedParams = dict() if embedParams is None else embedParams
    m = importlib.import_module(adr_path, package=package)
    for k in embedParams:
        setattr(m, k, embedParams[k])
    return m

def Import(fpath, reload=False, partialFlag=True, package=None, embedParams=None):
    if partialFlag: # e.g. fpath="articles.taming_transformers.taming.models.vqgan.VQModel"
        _module, _cls = fpath.rsplit('.', 1)
        if reload:
            module_imp = importlib.import_module(_module)
            importlib.reload(module_imp)
        return getattr(__imp(_module, package=package, embedParams=embedParams), _cls)
    else:
        return __imp(fpath, package=package, embedParams=embedParams)


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