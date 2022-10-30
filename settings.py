from os import getenv
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
    '-r',
    '--resume',
    type=str,
    const=True,
    default=pathBIO(getenv('GENIE_ML_R_DIR')),
    nargs='?',
    help='resume from logdir or checkpoint in logdir',
)
parser.add_argument(
    '-m',
    '--metrics',
    type=str,
    const=True,
    default=None,
    nargs='?',
    help='metrics table name',
)
opt, unknown = parser.parse_known_args()
RESUME = opt.resume
METRICS = opt.metrics

check_logdir(RESUME)
logs = readBIO(join(RESUME, 'config.yaml'), dotdictFlag=False)
LOGS_APP_NAME = logs['app']['name']

# load CONFIG
CONFIG = dotdict({
    'root': readBIO('//config.yaml', dotdictFlag=False),
    'models': readBIO(join('//apps', LOGS_APP_NAME, 'models', 'config.yaml'), dotdictFlag=False),
    'logs': logs,
    'ARGS': {'RESUME': RESUME, 'METRICS': METRICS}
})

#database handler
metrics = Metrics(
    f'//apps/{LOGS_APP_NAME}',
    'metrics',
    METRICS if METRICS else f'{CONFIG.logs.model.name}__{CONFIG.logs.data.name}__{getTimeHR(split="", dateFormat="%YY%mM%dD", timeFormat="%HH%MM%SS")}',
    CONFIG.logs.metrics
)