import os, tarfile, glob, shutil
from os import makedirs, system, environ, getenv, link, rename
from os.path import join, exists, relpath, getsize
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from libs.coding import sha1
from libs.basicIO import extractor, pathBIO, download
from libs.basicDS import retrieve

# from apps.VQGAN.util import retrieve

class ImageNetBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        self.HOST_DIR = self.config['HOST_DIR']
        if self.HOST_DIR.upper() == '$KAGGLE_PATH':
            self.HOST_DIR = pathBIO('//' + getenv('KAGGLE_PATH'))

        self.NAME = self.config['NAME']
        self.FILES = self.config.get('FILES', [])
        # self.root = join(cachedir, 'autoencoders/data', self.NAME)
        self.root = join(pathBIO(getenv('GENIE_ML_CACHEDIR')), self.NAME)
        self.datadir = join(self.root, 'data')
        self.hashdir = join(self.root, 'hash')
        makedirs(self.hashdir, exist_ok=True)
        self.txt_filelist = join(self.root, 'filelist.txt')
        
        self.preparation() # this function is overwrite by the user.

        if not self.bdu.is_prepared(self.root):
            logger.info('{} | Preparing dataset {} in {}'.format(self.__class__.__name__, self.NAME, self.root))
            datadir = self.datadir
            makedirs(datadir, exist_ok=True)
            for fname in self.FILES:
                fake_fpath = join(self.root, fname)
                if not exists(fake_fpath):
                    real_fdir = join(self.HOST_DIR, self.NAME)
                    real_fpath = join(real_fdir, fname)
                    real_fpath = (glob.glob(real_fpath + '*') + [real_fpath])[0]
                    if not exists(real_fpath):
                        self.download_dataset(real_fdir=real_fdir)
                        real_fpath = glob.glob(real_fpath + '*')[0]
                    
                    print('real_fpath', real_fpath)
                    print('fake_fpath', fake_fpath)
                    link(src=real_fpath, dst=fake_fpath)
                
                hashbased_path = join(self.hashdir, sha1(fake_fpath))
                if not exists(hashbased_path):
                    try:
                        makedirs(hashbased_path, exist_ok=True)
                        self.extract_dataset(fake_fpath=fake_fpath, datadir=datadir)
                    except Exception as e:
                        logger.error('@@@@@@@@@ e', e)

            filelist = glob.glob(join(datadir, '**', '*.{}'.format(self.config['EXT'])))
            filelist = [relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = '\n'.join(filelist) + '\n'
            with open(self.txt_filelist, 'w') as f:
                f.write(filelist)

            self.bdu.mark_prepared(self.root)
        
        self.df = pd.read_csv(self.df_path)

    def preparation(self, **kwargs):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths, cb=None, ignore_list=None):
        if exists(self.filtered_filelist):
            return np.load(self.filtered_filelist)

        # Example: ignore_list = ['n06596364_9591.JPEG', ...]
        ignore_list = ignore_list if ignore_list else []
        ignore = set(ignore_list)
        cb = cb if cb else lambda inp: True
        _cb = lambda _inp: bool(cb(_inp) and (not _inp.split('/')[-1] in ignore))
        relpaths = [rpath for rpath in tqdm(relpaths, desc='filtering of relpaths list') if _cb(rpath)]
        
        mode_val = self.config.get('MODE_VAL', None)
        if isinstance(mode_val, int):
            relpaths = [rpath for idx, rpath in tqdm(enumerate(relpaths), desc='filtering of relpaths list through indexing with mode={}'.format(mode_val)) if idx % mode_val == 0]

        np.save(self.filtered_filelist, relpaths)
        return relpaths
        if 'sub_indices' in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        self.human_dict = join(self.root, 'synset_human.txt')
        if not exists(self.human_dict):
            download(self.config['URL']['synset'], self.human_dict)

    def _prepare_idx_to_synset(self):
        self.idx2syn = join(self.root, 'index_synset.yaml')
        if not exists(self.idx2syn):
            download(self.config['URL']['iSynsetTest'], self.idx2syn)

    def _load(self):
        drGrade = lambda image_id_value: (list(self.df.loc[self.df['image_id']==image_id_value].dr) + [None])[0]
        cb = lambda inp: isinstance(drGrade(inp.split('/')[-1]), (int, float))
        
        with open(self.txt_filelist, 'r') as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths, cb=cb)
            logger.info('{} | ({}/{}) -> Removed {} files from filelist during filtering.'.format(self.__class__.__name__, len(self.relpaths), l1, l1 - len(self.relpaths)))

        if exists(self.synsets_of_filtered_filelist):
            self.synsets = np.load(self.synsets_of_filtered_filelist)
        else:
            self.synsets = ['class_' + str(drGrade(p.split('/')[-1])) for p in tqdm(self.relpaths, desc='creation of synsets list')]
            np.save(self.synsets_of_filtered_filelist, self.synsets)
        logger.info('{} | relpaths len: {}, Synset len: {}'.format(self.__class__.__name__, len(self.relpaths), len(self.synsets)))
        self.abspaths = [join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        logger.info('{} | unique_synsets: {}'.format(self.__class__.__name__, unique_synsets))
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        self.class_labels = [class_dict[s] for s in self.synsets]
        logger.info('{} | class_dict: {}'.format(self.__class__.__name__, class_dict))

        with open(self.human_dict, 'r') as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        print('XXXXXX human_dict XXXXXXX', human_dict)
        self.human_labels = [human_dict[s] for s in self.synsets]
        # synset and human_labels logicly is equal, they're used for machine and human respectivly.

        labels = {
            'relpath': np.array(self.relpaths),
            'synsets': np.array(self.synsets),
            'class_label': np.array(self.class_labels),
            'human_label': np.array(self.human_labels),
        }
        self.data = self.ImagePaths(
            self.abspaths,
            labels=labels,
            size=retrieve(self.config, 'SIZE', default=0),
            random_crop=self.random_crop
        )

class ImageNetTrain(ImageNetBase):
    """
    it useful to overide functions below in creation of custom dataset:
        `download_dataset`, `extract_dataset`, `preparation of parrent class`
    """
    def download_dataset(self, **kwargs):
        raise NotImplementedError()
    
    def extract_dataset(self, **kwargs):
        fake_fpath = kwargs['fake_fpath']
        datadir = kwargs['datadir']
        extractor(src_file=fake_fpath, dst_dir=datadir, mode='zip')
        nested_list = glob.glob(join(datadir, '*.zip*'))
        assert len(nested_list)==0, f'nested_list: {nested_list} is exist.'

class ImageNetValidation(ImageNetBase):
    """
    it useful to overide functions below in creation of custom dataset:
        `download_dataset`, `extract_dataset`, `preparation of parrent class`
    """
    def download_dataset(self, **kwargs):
        raise NotImplementedError()
    
    def extract_dataset(self, **kwargs):
        fake_fpath = kwargs['fake_fpath']
        datadir = kwargs['datadir']
        extractor(src_file=fake_fpath, dst_dir=datadir, mode='zip')
        nested_list = glob.glob(join(datadir, '*.zip*'))
        assert len(nested_list)==0, f'nested_list: {nested_list} is exist.'