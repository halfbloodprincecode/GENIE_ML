import yaml
import json
import glob
import pathlib
from os.path import join, exists
from libs.basicHR import EHR
from libs.basicDS import dict2ns, dotdict

def rootPath():
    return pathlib.Path(__file__).parents[1]

def pathBIO(fpath: str, **kwargs):
    if fpath.startswith('//'):
        fpath = join(rootPath(), fpath[2:])
    return fpath

def readBIO(fpath: str, **kwargs):
    """
        yamlFile = readBIO('/test.yaml')
        jsonFile = readBIO('/test.json')
    """
    fpath = pathBIO(fpath)
    ext = fpath.split('.')[-1].lower()
    dotdictFlag = kwargs.get('dotdictFlag', None)

    if ext == 'yaml':
        try:
            with open(fpath, 'r') as f:
                return dotdict(yaml.safe_load(f), flag=dotdictFlag)
        except Exception as e:
            EHR(e)
    
    if ext == 'json':
        try:
            with open(fpath) as f:
                return dotdict(json.load(f), flag=dotdictFlag)
        except Exception as e:
            EHR(e)

def check_logdir(fpath: str, **kwargs):
    fpath = pathBIO(fpath)
    if not exists(fpath): # TODO AND fpath INSIDE LOG DIR.
        pass 

def ls(_dir, _pattern: str, full_path=False):
    """
        print(glob.glob('/home/adam/*.txt'))   # All files and directories ending with .txt and that don't begin with a dot:
        print(glob.glob('/home/adam/*/*.txt')) # All files and directories ending with .txt with depth of 2 folders, ignoring names beginning with a dot:
    Example:
        print(ls('/', '*.jpg'))
    """
    if isinstance(_dir, dict):
        return list(_dir.keys()) # _dir contains multi directory informations.
    assert isinstance(_dir, str), '_dir is must be str (path directory)'
    
    _dir = pathBIO(_dir)

    if full_path:
        return glob.glob(join(_dir, _pattern))
    else:
        return glob.glob1(_dir, _pattern)