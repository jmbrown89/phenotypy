import sys
from appdirs import AppDirs
from pkg_resources import get_distribution, DistributionNotFound
from argparse import ArgumentParser

from phenotypy.misc.utils import parse_config, parse_config_ls
from phenotypy.models.train_model import train

try:
    __version__ = get_distribution('phenotypy').version
except DistributionNotFound:
    __version__ = None


class Phenotypy(object):

    def __init__(self):

        parser = ArgumentParser(
            description='A set of commands for image and video based phenotyping',
            usage='''phenotypy <command> [<args>]

            The following commands are available:
               update   Download the latest models, specific versions, or configure auto-update
               train    Train a new model for HCA using a custom configuration file
               config   Generate a configuration file for your data and requirements
            ''')

        parser.add_argument('command', help='Command to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def update(self):

        parser = ArgumentParser(
            description='Download the latest models, specific versions, or configure auto-update')

        parser.add_argument('update')

        sp = parser.add_subparsers()
        latest = sp.add_parser('latest', help='Download the latest models')
        auto = sp.add_parser('auto', help='Toggle automatic model updates')
        version = sp.add_parser('version', help='Download specific model version')

    def train(self):

        parser = ArgumentParser(description='Train a new model for HCA using a custom configuration file')

        parser.add_argument('-c', '--config', help='Configuration file', dest='config', required=True)
        parser.add_argument('--search', help='Perform line search', action='store_true', dest='search', default=False)
        args = parser.parse_args(sys.argv[2:])

        if args.search:
            from phenotypy.models.line_search import line_search
            line_search(args.config)
        else:
            from phenotypy.models.train_model import train
            train(args.config)


def initialize():

    # Setup appdirs
    dirs = AppDirs("phenotypy", "LoVE", version=__version__)
    conf_dir = dirs.user_config_dir
    return conf_dir


def main():
    Phenotypy()
