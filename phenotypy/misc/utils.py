import yaml
from pathlib import Path
import logging


def init_log(out_path):

    log_fmt = '%(asctime)s, %(name)s, %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, filename=out_path, filemode='w')


def increment_path(directory, name_pattern):
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


def parse_config(config_file):

    config = yaml.load(open(config_file, 'r').read())
    config['data_dir'] = str(Path.resolve(Path(config_file).parent / config['data_dir']))

    if not config.get('out_dir', None):

        out_dir = (Path.home() / 'phenotypy_out')
        try:
            out_dir.mkdir(parents=False, exist_ok=True)
        except FileNotFoundError:
            logging.error(f"Unable to create output directory '{out_dir}'. "
                          f"Please specify a valid 'out_dir' entry in your config file")
            exit(1)
    else:
        out_dir = Path.resolve(Path(config_file).parent / config['out_dir'])
        out_dir.mkdir(parents=False, exist_ok=True)

    config['out_dir'] = str(out_dir)
    return config


def parse_config_ls(config_file):

    config = parse_config(config_file)
    params = [key for key, value in config.items() if isinstance(value, list)]
    return config, params
