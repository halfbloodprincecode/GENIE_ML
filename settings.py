from os import getenv
from libs.args import Args
from dotenv import load_dotenv
from libs.time import getTimeHR
from utils.metrics import Metrics
from os.path import dirname, join
from setuptools import find_packages, setup

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

# load constant args(configs)
import args
import models.args 
CONFIG = Args.export()

#database handler
import database as my_db
metrics = Metrics(my_db, CONFIG.METRICS_DB, CONFIG.METRICS_TBL if CONFIG.METRICS_TBL else f'{CONFIG.MODEL}_{CONFIG.DATASET}_{getTimeHR(split="", dateFormat="%YY%mM%dD", timeFormat="%HH%MM%SS")}', CONFIG.METRICS)

print(CONFIG)