import click
import logging
from pathlib import Path, PurePath
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import yaml
import numpy as np

import torch
from torch import optim
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint

from phenotypy.data.make_dataset import parse_config, create_data_loaders
from phenotypy.misc.utils import get_logger, get_experiment_dir
from phenotypy.models import resnet
from phenotypy.visualization.plotting import Plotter


@click.command()
@click.argument('config', type=click.Path(exists=True))
@click.option('--cv', is_flag=True)
def main(config, cv):
    """ Runs training based on the provided config file.
    """
    cross_validate(config) if cv else train(config)


def cross_validate(config_path):

    config = parse_config(Path(config_path))
    out_dir = Path(config['out_dir'])
    data_dir = Path(config['data_dir'])
    cv_dir = data_dir / 'cross_validation'
    cv_file = config.get('cross_validation', None)

    try:
        cross_val_path = data_dir / cv_file
        if not Path(cross_val_path).exists():
            raise FileNotFoundError()

    except TypeError:
        print("Please specify a valid CSV file for cross-validation.")
        exit(1)
    except FileNotFoundError:
        print(f"Cross-validation file '{cv_file}' not found in {data_dir}.")
        exit(1)

    cv_dir.mkdir(parents=False, exist_ok=config['clobber'])

    video_list = pd.read_csv(data_dir / cross_val_path)['video'].apply(lambda x: data_dir / x).values

    for validation in range(len(video_list)):

        config_path = (out_dir / f'split_{validation}').with_suffix('.yaml')

        if config_path.exists() and (out_dir / f'split_{validation}' / 'final_model.pth').exists():
            print(f'CV split {validation} already complete. Skipping...')
            continue

        train_list = [video_list[training] for training in range(len(video_list)) if training != validation]
        validation_list = [video_list[validation]]

        train_csv = cv_dir / f'training_{validation}.csv'
        val_csv = cv_dir / f'validation_{validation}.csv'

        config['training_csv'] = str(Path(*train_csv.parts[-2:]))
        config['validation_csv'] = str(Path(*val_csv.parts[-2:]))
        pd.DataFrame(pd.Series(train_list, name='video')).to_csv(train_csv)
        pd.DataFrame(pd.Series(validation_list, name='video')).to_csv(val_csv)

        with open(config_path, 'w') as conf:
            yaml.dump(config, conf, default_flow_style=False)

        train(config_path)


def train(config_path, experiment_name=None):

    config = parse_config(Path(config_path))
    seed = config.get('seed', 1984)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    if not experiment_name:
        experiment_name = Path(config_path).stem

    experiment_dir = get_experiment_dir(config, experiment_name)
    log_file = (experiment_dir / experiment_name).with_suffix('.log')
    print(f"Logging to '{log_file}'")
    logger = get_logger(experiment_name, log_file)
    logger.info('Training started')

    config['experiment_dir'] = str(experiment_dir)
    train_loader, val_loader = create_data_loaders(config)
    train_plotter = Plotter(experiment_dir, vis=config['visdom'])
    val_plotter = Plotter(experiment_dir, vis=config['visdom'])

    with open(experiment_dir / Path(config_path).name, 'w') as f:
        config['encoding'] = train_loader.dataset.label_encoding
        yaml.dump(config, f, default_flow_style=False)

    logger.info("Initialising model")
    layers = config['layers']
    model = resnet.resnet[layers](  # https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md#shortcuttype
        num_classes=config['no_classes'],
        sample_size=config['transform']['resize'],
        sample_duration=config['clip_length'],
        init=config.get('weights', 'xavier'))

    params = resnet.get_fine_tuning_parameters(model)
    solver = config.get('solver', 'sgd').lower()

    if 'rms' in solver:
        optimizer = optim.RMSprop(params, lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    elif 'sgd' in solver:
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.Adam(params, amsgrad=True)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

    weight_tensor = None
    if config.get('class_weights'):
        class_weights = train_loader.dataset.get_class_weights()
        logger.info(f'Using class weights: {class_weights}')
        weight_tensor = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())]).to(device)

    loss = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    trainer = create_supervised_trainer(model, optimizer, loss, device=device, non_blocking=True)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(loss)},
                                            device=device, non_blocking=True)

    train_loss, val_loss, val_acc = [], [], []
    try:
        checkpointer = ModelCheckpoint(Path(experiment_dir) / 'checkpoints', 'checkpoint',
                                       save_interval=1, n_saved=config['epochs'],
                                       require_empty=not config['clobber'])
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model})
    except ValueError:
        logger.warning("Unable to save checkpoints - either delete old models or pass 'clobber = True' in config.")

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):

        iteration = (engine.state.iteration - 1) % len(train_loader) + 1

        if iteration % config['log_interval'] == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                        .format(engine.state.epoch, iteration, len(train_loader), engine.state.output))
            train_plotter.plot_loss(engine)
            train_loss.append(engine.state.output)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        # scheduler.step(avg_loss)

        val_acc.append(avg_accuracy)
        val_loss.append(avg_loss)
        logger.info("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                     .format(engine.state.epoch, avg_accuracy, avg_loss))
        val_plotter.plot_loss_accuracy(engine, avg_accuracy, avg_loss)

    trainer.run(train_loader, max_epochs=config['epochs'])
    logger.info("Training complete!")

    # Results
    torch.save(model, str(Path(experiment_dir) / 'final_model.pth'))
    results = pd.DataFrame()
    results['accuracy'] = val_acc
    results['loss'] = val_loss
    results.to_csv(Path(experiment_dir / 'val_results.csv'))
    pd.DataFrame(pd.Series(train_loss, name='loss')).to_csv(Path(experiment_dir / 'train_results.csv'))
    logger.info(f"Results saved to '{experiment_dir}'")
    return results


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
