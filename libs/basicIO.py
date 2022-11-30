import os
import yaml
import json
import glob
import pathlib
import tarfile
import zipfile
import requests
from PIL import Image
from tqdm import tqdm
from loguru import logger
from libs.coding import md5
from libs.basicHR import EHR
from os.path import join, exists
from os import getenv, symlink, link
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
        with tarfile.open(src_file, 'r:') as tar_ref:
            # tar_ref.extractall(path=dst_dir)
            for file in tqdm(iterable=tar_ref.namelist(), total=len(tar_ref.namelist()), desc='extracting {}'.format(src_file)):
                # Extract each file to another directory
                # If you want to extract to current working directory, don't specify path
                tar_ref.extract(member=file, path=dst_dir)
    
    if mode == 'zip':
        with zipfile.ZipFile(src_file, 'r') as zip_ref:
            # zip_ref.extractall(dst_dir)
            for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), desc='extracting {}'.format(src_file)):
                # Extract each file to another directory
                # If you want to extract to current working directory, don't specify path
                zip_ref.extract(member=file, path=dst_dir)

def download(url: str, local_path, chunk_size=1024):
    if exists(local_path):
        if getenv('GENIE_ML_DEBUG_MODE') == 'True':
            logger.debug('local_path already is exist.')
        return

    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    if not url.startswith('http'):
        url = pathBIO(url)
        link(src=url, dst=local_path)
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

def signal_save(s, path, makedirsFlag=True):
    path = pathBIO(path)
    d, ext = os.path.split(path)
    if makedirsFlag:
        os.makedirs(d, exist_ok=True)
    logger.error('###### type(s)={} | ext={}'.format(type(s), ext))
    # Image.fromarray(grid).save(path)