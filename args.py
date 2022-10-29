from libs.args import Args

args = Args('')

# args.add_argument('tEst', type=str, default='t1')
args.add_argument('o', 'outdir', type=str, default=None)
args.add_argument('m', 'metrics', type=str, default=None)
args.add_argument('r', 'resume', type=str, default='//logs/vqgan/firstStage-IdrId')
args.add_argument('t', 'train', type=Args.str2bool, const=True, default=False, nargs='?')

# args.add_argument( 
#     'b',
#     'base',
#     nargs='*',
#     metavar='base_config.yaml',
#     help='paths to base configs. Loaded from left-to-right.'
#     'Parameters can be overwritten or added with command-line options of the form `--key value`.',
#     default=list()
# )

# args.add_argument(
#     's',
#     'seed',
#     type=int,
#     default=23,
#     help='seed for seed_everything',
# )

args = args.parser()