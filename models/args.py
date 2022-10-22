from libs.args import Args

args = Args('vqgan')

args.add_argument('test', type=str, default='test')

args = args.parser()