from loguru import logger
from os import getenv, environ, makedirs
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
# parser.add_argument(
#     '-m',
#     '--metrics_tbl',
#     type=str,
#     const=True,
#     default=None,
#     nargs='?',
#     help='metrics table name',
# )
opt, unknown = parser.parse_known_args()

environ['GENIE_ML_APP'] = opt.app

import kaggle # need to import here(after env variables had defined)
makedirs(environ['KAGGLE_PATH'], exist_ok=True)
kaggle.api.CONFIG_NAME_PATH = 'KAGGLE_PATH'.lower()
print('####################3', environ['KAGGLE_PATH'])
kaggle.api.authenticate()
print(kaggle.api.CONFIG_NAME_PATH)

#database handler
# metrics = Metrics(
#     f'//apps/{opt.app}',
#     'metrics',
#     opt.metrics_tbl if opt.metrics_tbl else f'{CONFIG.logs.model.name}__{CONFIG.logs.data.name}__{getTimeHR(split="", dateFormat="%YY%mM%dD", timeFormat="%HH%MM%SS")}',
#     CONFIG.logs.metrics
# )


# environ['GENIE_ML_METRICS'] = None

# logs = readBIO(join(RESUME, 'config.yaml'), dotdictFlag=False)
# LOGS_APP_NAME = logs['app']['name']

# # load CONFIG
# CONFIG = dotdict({
#     'root': readBIO('//config.yaml', dotdictFlag=False),
#     'models': readBIO(join('//apps', LOGS_APP_NAME, 'models', 'config.yaml'), dotdictFlag=False),
#     'logs': logs,
#     'ARGS': {'RESUME': RESUME, 'METRICS': METRICS}
# })

