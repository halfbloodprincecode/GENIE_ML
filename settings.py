import os
import shutil
from loguru import logger
from os import getenv, environ, makedirs, system, sep
from dotenv import load_dotenv
from libs.basicDS import dotdict
from utils.metrics import Metrics
from os.path import dirname, join, exists
from argparse import ArgumentParser
from libs.basicTime import getTimeHR
from setuptools import find_packages, setup
from libs.basicIO import pathBIO, readBIO, check_logdir

#load env variables.
load_dotenv(join(dirname(__file__), '.env'))

assert environ['GENIE_ML_STORAGE0'].endswith(sep + '@GENIE_ML_APP')

# setup the project
if getenv('SETUP') == 'True':
    setup(
        name=getenv('NAME'),
        packages=find_packages(),
        version=getenv('VERSION'),
        description=getenv('DESCRIPTION'),
        author=getenv('AUTHOR'),
        license=getenv('LICENSE'),
    )

# load some of ARGS
parser = ArgumentParser()
parser.add_argument(
    '--app',
    type=str,
    required=True,
    help='app name',
)
parser.add_argument(
    '--app-fn',
    type=str,
    default='main',
    help='app function name',
)
opt, unknown = parser.parse_known_args()

app_splited = opt.app.split(':')
app_module = app_splited[0]

if len(app_splited) == 1 and len(app_splited[0]) > 0:
    environ['GENIE_ML_APP'] = app_splited[0]
elif len(app_splited) == 2 and len(app_splited[0]) > 0 and len(app_splited[1]) > 0:
    environ['GENIE_ML_APP'] = app_splited[1]
else:
    raise ValueError('app name `{}` is not valid'.format(opt.app))

environ['GENIE_ML_APP_FN'] = opt.app_fn

PFX_KEYS = environ['GENIE_ML_PFX'].split(',')

for k, v in environ.items():
    flag = False
    for valid_prefix in PFX_KEYS:
        if k.startswith(valid_prefix):
            flag = True
            break
    
    if not flag:
        continue
    
    new_v = []
    for vi in v.split(sep):
        if vi.startswith('@'):
            new_v.append(environ[vi[1:]])
        else:
            new_v.append(vi)
    new_v = sep.join(new_v)
    environ[k] = new_v

# if bool(environ['GENIE_ML_APP_DSC']) and (not exists(environ['GENIE_ML_STORAGE0'])) and exists(os.path.split(environ['GENIE_ML_STORAGE0'])[0]):
#     shutil.copytree(os.path.split(environ['GENIE_ML_STORAGE0'])[0], environ['GENIE_ML_STORAGE0'])

# https://github.com/Kaggle/kaggle-api
if getenv('KAGGLE_CHMOD'):
    system('chmod {} {}'.format(
        str(getenv('KAGGLE_CHMOD')),
        join(str(getenv('KAGGLE_CONFIG_DIR')), 'kaggle.json')
    ))

import kaggle # need to import here(after env variables had defined)