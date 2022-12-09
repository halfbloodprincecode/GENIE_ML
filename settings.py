from loguru import logger
from os import getenv, environ, makedirs, system, sep
from dotenv import load_dotenv
from libs.basicDS import dotdict
from utils.metrics import Metrics
from os.path import dirname, join
from argparse import ArgumentParser
from libs.basicTime import getTimeHR
from setuptools import find_packages, setup
from libs.basicIO import pathBIO, readBIO, check_logdir

#load env variables.
load_dotenv(join(dirname(__file__), '.env'))

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
opt, unknown = parser.parse_known_args()

environ['GENIE_ML_APP'] = opt.app
for k, v in environ.items():
    new_k = []
    for key in k.split(sep):
        if key.startswith('@'):
            new_k.append(environ[key[1:]])
        else:
            new_k.append(key)
    new_k = sep.join(new_k)
    environ[new_k] = v

# https://github.com/Kaggle/kaggle-api
if getenv('KAGGLE_CHMOD'):
    system('chmod {} {}'.format(
        str(getenv('KAGGLE_CHMOD')),
        join(str(getenv('KAGGLE_CONFIG_DIR')), 'kaggle.json')
    ))

import kaggle # need to import here(after env variables had defined)