import os
import yaml
import json
import glob
import pathlib
import requests
from tqdm import tqdm
from os import symlink
from os.path import join, exists
from libs.coding import md5
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
    assert isinstance(_dir, str), f'_dir is must be str (path directory). but now is {type(_dir)}'
    
    _dir = pathBIO(_dir)

    if full_path:
        return glob.glob(join(_dir, _pattern))
    else:
        return glob.glob1(_dir, _pattern) # Notic: glob1 does not support (**) (match child dirs)

def is_prepared(adr):
    return pathlib.Path(adr).joinpath('.ready').exists()

def mark_prepared(adr):
    pathlib.Path(adr).joinpath('.ready').touch()

def extractor(src_file, dst_dir, mode='tar'):
    if mode == 'tar':
        import tarfile
        with tarfile.open(src_file, 'r:') as tar:
            tar.extractall(path=dst_dir)
    
    if mode == 'zip':
        import zipfile
        with zipfile.ZipFile(src_file, 'r') as zip_ref:
            zip_ref.extractall(dst_dir)

def download(url: str, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    if not url.startswith('http'):
        url = pathBIO(url)
        symlink(src=url, dst=local_path)
        return

    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            with open(local_path, 'wb') as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def file_hash(path, fn=md5):
    with open(path, 'rb') as f:
        content = f.read()
    return fn(content)

