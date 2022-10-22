from libs.args import Args

args = Args('')

args.add_argument('tEst', type=str, default='t1')
args.add_argument('model', type=str, default='mlp')
args.add_argument('dataset', type=str, default='D')
args.add_argument('metrics_db', type=str, default='metrics')
args.add_argument('metrics_tbl', type=str, default=None)
args.add_argument('metrics', type=lambda s: [str(item).strip() for item in s.split(',') if str(item).strip()], default='loss,val_loss')

args = args.parser()