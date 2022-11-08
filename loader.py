import pretty_errors
from os import getenv
# from loguru import logger
import settings
# from libs.basicIO import readBIO, ls
from libs.dyimport import Import


app = Import(f'apps.{getenv("GENIE_ML_APP")}.app.main', embedParams={})

# from articles.Attention_is_All_You_Need.index import main as atn_main, metrics as atn_metrics
# from utils.quantizer import veqQuantizerImg
# from utils.loss import vqvae_loss

# print(ARGS, getenv('NAME'))