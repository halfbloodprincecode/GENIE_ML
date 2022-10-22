from os import getenv
from libs.args import Args
from dotenv import load_dotenv
from argparse import Namespace
from libs.basicDS import ns_add
from os.path import dirname, join
from setuptools import find_packages, setup

load_dotenv(join(dirname(__file__), '.env'))

if getenv('SETUP') == 'True':
    setup(
        name=getenv('NAME'),
        packages=find_packages(),
        version=getenv('VERSION'),
        description=getenv('DESCRIPTION'),
        author=getenv('AUTHOR'),
        license=getenv('LICENSE'),
    )

import args
import models.args 
CONFIG = Args.export()