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
    if not k.startswith('GENIE_'):
        continue
    if not k == 'GENIE_ML_LOGDIR':
        continue
    new_v = []
    for vi in v.split(sep):
        print(vi)
        if vi.startswith('@'):
            new_v.append(environ[vi[1:]])
        else:
            new_v.append(vi)
    new_v = sep.join(new_v)
    environ[k] = new_v
    print('k={} | getenv={} | environ={}'.format(k, getenv(k), environ[k]))

# https://github.com/Kaggle/kaggle-api
if getenv('KAGGLE_CHMOD'):
    system('chmod {} {}'.format(
        str(getenv('KAGGLE_CHMOD')),
        join(str(getenv('KAGGLE_CONFIG_DIR')), 'kaggle.json')
    ))

import kaggle # need to import here(after env variables had defined)