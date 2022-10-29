from libs.args import Args

print('hooooooooooooooO!!')

args = Args('vqgan')

args.add_argument('test', type=str, default='test')
args.add_argument('resolution', type=int, default=256)
args.add_argument('conditional', type=str, default='objects_bbox')
args.add_argument('n', 'n_samples_per_layout', type=int, default=4)


## ** Not Needed (this information is provided in CONFIG.logs) **
# args.add_argument('model', type=str, default=None)
# args.add_argument('dataset', type=str, default=None)
# args.add_argument('metrics', type=lambda s: [str(item).strip() for item in s.split(',') if str(item).strip()], default='loss,val_loss')

args = args.parser()