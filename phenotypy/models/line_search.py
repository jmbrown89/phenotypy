import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from copy import deepcopy
import yaml
import torch
import numpy

from phenotypy.misc.utils import parse_config_ls, init_log
from phenotypy.misc.math import mrange
from phenotypy.models.train_model import train


@click.command()
@click.argument('config', type=click.Path(exists=True))
def main(config):
    """ Runs training based on the provided config file.
    """
    line_search(config)


def line_search(config):

    config_dict, params = parse_config_ls(Path(config))

    # Set random seed
    rs = config_dict.get('seed', 1984)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(rs)
    numpy.random.seed(rs)

    for search_param in sorted(params):

        search_config = deepcopy(config_dict)
        search_range = None

        for fixed in params:  # set the fixed value for all other parameters

            if fixed == 'layers':
                fixed_range = config_dict[fixed]
                fixed_value = 34
            else:
                fixed_range = mrange(*config_dict[fixed])
                fixed_value = fixed_range[len(fixed_range) // 2]

            if search_param != fixed:
                search_config[fixed] = fixed_value
            else:
                search_range = fixed_range

        for search_val in search_range:

            search_config[search_param] = search_val
            search_config['base_config'] = str(Path(config).absolute())
            out_file = Path(config_dict['out_dir']) / f'{search_param}_{search_val}.yaml'

            with open(out_file, 'w') as f:
                yaml.dump(search_config, stream=f, default_flow_style=False)  # output the config for reproducibility

            val_results = train(out_file)


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
