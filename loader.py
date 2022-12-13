import pretty_errors
from os import getenv
# from loguru import logger
from settings import app_module
# from libs.basicIO import readBIO, ls
from libs.dyimport import Import


app = Import(f'apps.{app_module}.app.App', embedParams={})

# from articles.Attention_is_All_You_Need.index import main as atn_main, metrics as atn_metrics
# from utils.quantizer import veqQuantizerImg
# from utils.loss import vqvae_loss

# print(ARGS, getenv('NAME'))