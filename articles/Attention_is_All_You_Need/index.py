from . import database
from . import architecture
from utils.sqlite import Metrics

metrics = Metrics(database, 'my-db', [
    'loss', 
    'val_loss'
])

def main():
    print('ok', architecture)