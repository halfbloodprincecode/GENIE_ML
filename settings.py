from os import getenv
from loguru import logger
from libs.args import Args
from argparse import Namespace
from dotenv import load_dotenv
from libs.time import getTimeHR
from libs.basicDS import dotdict
from libs.dyimport import Import
from utils.metrics import Metrics
from os.path import dirname, join
from setuptools import find_packages, setup
from libs.basicIO import readBIO, check_logdir

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

# load constant ARGS
import args 
RESUME = getattr(Args.export(), 'RESUME', None)
check_logdir(RESUME)
logs = readBIO(join(RESUME, 'config.yaml'))
Import(f'apps.{logs.app.name}.models.args', partialFlag=False)
ARGS = Args.export()

# load constant CONFIG
CONFIG = dotdict({
    'root': readBIO('//config.yaml'),
    'models': readBIO(f'//apps/{logs.app.name}/models/config.yaml'),
    'logs': logs
})

#database handler
metrics = Metrics(
    f'//apps/{logs.app.name}',
    'metrics',
    ARGS.METRICS if ARGS.METRICS else f'{CONFIG.logs.model.name}__{CONFIG.logs.data.name}__{getTimeHR(split="", dateFormat="%YY%mM%dD", timeFormat="%HH%MM%SS")}',
    CONFIG.logs.metrics
)

# logger.info(ARGS)
# logger.info(CONFIG.models)