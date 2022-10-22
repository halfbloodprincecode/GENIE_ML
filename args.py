from libs.args import Args

args = Args('')

args.add_argument('test', type=str, default='t1')
args.add_argument('model', type=str, default='mlp')
args.add_argument('dataset', type=str, default='D')

args = args.parser()