import argparse
import os
from pathlib import Path

class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        data = parser.add_argument_group('Data')
        data.add_argument('--dir_data', type=str, default='data/cifar10', help='path to images (data folder)')
        data.add_argument('--dir_results', type=str, default='result/logs', help='path to results')
        data.add_argument('--dir_checkpoints', type=str, default='result/checkpoints', help='path to results')
        
        exp = parser.add_argument_group('Experiment')
        exp.add_argument('--name', type=str, default='Curriculum', help='name of the experiment. It decides where to store samples and models')
        exp.add_argument('--server', default=False)
        exp.add_argument('--model', default=False)
        exp.add_argument('--epoch', default=100000)

        buf = parser.add_argument_group('Buffer')
        # buf.add_argument('--seed_buffer_size', default=128)
        buf.add_argument('--batch_size', default=16)
        buf.add_argument('--sample_size', default=128)
        buf.add_argument('--test_size', default=512)
        buf.add_argument('--augment_num', default=2)
        buf.add_argument('--init_buffer_size', default=3000)
        buf.add_argument('--score_metric', default='error', help='error or std')
        
        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        Path(opt.dir_results).mkdir(parents=True, exist_ok=True)

        Path(opt.dir_checkpoints).mkdir(parents=True, exist_ok=True)

        return opt
