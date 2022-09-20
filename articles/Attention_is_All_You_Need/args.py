from argparse import ArgumentParser
f = lambda key: '--' + 'atn' + '-' + key

args = ArgumentParser()
args.add_argument(f('model'), type=str, default='mlp')
args.add_argument(f('dataset'), type=str, default='D')
args = args.parse_known_args()[0]