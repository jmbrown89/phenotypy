import yaml
from pathlib import Path
import logging

loggers = {}
log_fmt = '%(asctime)s, %(name)s, %(levelname)s: %(message)s'


def get_logger(name, out_path=None):

    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        handler = logging.FileHandler(out_path)
        formatter = logging.Formatter(log_fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        loggers[name] = logger
        return logger


def init_log(out_path):

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


def get_experiment_dir(config, experiment_name):

    try:
        experiment_dir = Path(config['out_dir']) / experiment_name
        experiment_dir.mkdir(parents=False, exist_ok=config.get('clobber', False))
    except FileExistsError:
        experiment_dir = increment_path(Path(config['out_dir']), experiment_name + '_({})')
        experiment_dir.mkdir(parents=False, exist_ok=False)
        print(f"Warning: experiment '{experiment_name}' already exists!")

    return experiment_dir
