import os
from os import getenv
from libs.basicIO import ls, pathBIO
from omegaconf import OmegaConf
from libs.args import ParserBasic
from pytorch_lightning import seed_everything
from os.path import join, exists, isfile, isdir
from argparse import ArgumentParser, ArgumentTypeError

def _get_parser_(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise ArgumentTypeError('Boolean value expected.')

    parser = ArgumentParser(**parser_kwargs)
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        const=True,
        default='',
        nargs='?',
        help='postfix for logdir',
    )
    parser.add_argument(
        '-r',
        '--resume',
        type=str,
        const=True,
        default='',
        nargs='?',
        help='resume from logdir or checkpoint in logdir',
    )
    parser.add_argument(
        '-b',
        '--base',
        nargs='*',
        metavar='base_config.yaml',
        help='paths to base configs. Loaded from left-to-right. '
        'Parameters can be overwritten or added with command-line options of the form `--key value`.',
        default=list(),
    )
    parser.add_argument(
        '-t',
        '--train',
        type=str2bool,
        const=True,
        default=False,
        nargs='?',
        help='train',
    )
    parser.add_argument(
        '--no-test',
        type=str2bool,
        const=True,
        default=False,
        nargs='?',
        help='disable test',
    )
    parser.add_argument('-p', '--project', help='name of new or path to existing project')
    parser.add_argument(
        '-d',
        '--debug',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='enable post-mortem debugging',
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=23,
        help='seed for seed_everything',
    )
    parser.add_argument(
        '-f',
        '--postfix',
        type=str,
        default='',
        help='post-postfix for default name',
    )

    return parser

def _ctl_parser_(opt, unknown, **kwargs):
    now = kwargs['now']
    if opt.name and opt.resume:
        raise ValueError(
            '-n/--name and -r/--resume cannot be specified both.'
            'If you want to resume training in a new log folder, '
            'use -n/--name in combination with --resume_from_checkpoint'
        )
    if opt.resume:
        if not exists(opt.resume):
            raise ValueError('Cannot find {}'.format(opt.resume))
        if isfile(opt.resume): # ckpt address
            paths = opt.resume.split('/')
            idx = len(paths)-paths[::-1].index('logs')+1 # this can produce error if `logs` is not in list.
            logdir = '/'.join(paths[:idx])
            ckpt = opt.resume
        else: # logdir address
            assert isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip('/')
            ckpt = join(getenv('GENIE_ML_CKPTDIR') or join(logdir, 'checkpoints'), 'last.ckpt')

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(
            ls(logdir, 'configs/*.yaml', full_path=True)
        )


        aa = ls(logdir, 'configs/*.yaml', full_path=True)
        bb = sorted(aa)
        print('*'*30)
        print('aa', aa)
        print('bb', bb)

        opt.base = base_configs + opt.base
        _tmp = logdir.split('/')
        nowname = _tmp[_tmp.index('logs')+1]
    else:
        if opt.name:
            name = '_' + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = '_' + cfg_name
        else:
            name = ''
        nowname = now + name + opt.postfix
        logdir = join(pathBIO(getenv('GENIE_ML_LOGDIR')), nowname)

    ckptdir = getenv('GENIE_ML_CKPTDIR') or join(logdir, 'checkpoints')
    cfgdir = getenv('GENIE_ML_CFGDIR') or join(logdir, 'configs')
    seed_everything(opt.seed)
    return ckptdir, cfgdir, logdir, nowname

class Parser(ParserBasic):
    @classmethod
    def get_parser(cls, **kwargs):
        super().get_parser(**kwargs)
        parser_kwargs = kwargs.get('parser_kwargs', dict())
        return _get_parser_(**parser_kwargs)
    
    @classmethod
    def ctl_parser(cls, opt, unknown, **kwargs):
        super().ctl_parser(opt, unknown, **kwargs)
        return _ctl_parser_(opt, unknown, **kwargs)