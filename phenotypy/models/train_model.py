import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

import torch
from torch import optim
import torch.utils.data as data
import torch.nn.functional as F
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint

from phenotypy.data.make_dataset import parse_config, create_data_loaders
from phenotypy.misc.utils import init_log, increment_path
from phenotypy.models import resnet
from phenotypy.models.evaluate_model import confusion
from phenotypy.visualization.plotting import Plotter


@click.command()
@click.argument('config', type=click.Path(exists=True))
def main(config):
    """ Runs training based on the provided config file.
    """
    train(config)


def train(config_path, experiment_name=None):

    config = parse_config(Path(config_path))
    logger = logging.getLogger(__name__)

    if not experiment_name:
        experiment_name = Path(config_path).stem

    try:
        experiment_dir = Path(config['out_dir']) / experiment_name
        experiment_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        experiment_dir = increment_path(Path(config['out_dir']), experiment_name + '_({})')
        experiment_dir.mkdir(parents=False, exist_ok=False)
        print(f"Warning: experiment '{experiment_name}' already exists!")

    log_file = (experiment_dir / experiment_name).with_suffix('.log')
    print(f"Logging to '{log_file}'")
    init_log(log_file)
    logger.info('Training started')

    train_loader, val_loader = create_data_loaders(config)
    train_plotter = Plotter(experiment_dir, vis=config['visdom'])
    val_plotter = Plotter(experiment_dir, vis=config['visdom'])

    logger.info("Initialising model")
    layers = config['layers']
    model = resnet.resnet[layers](  # https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md#shortcuttype
        num_classes=config['no_classes'],
        sample_size=config['transform']['scale'],
        sample_duration=config['clip_length'])

    params = resnet.get_fine_tuning_parameters(model)
    optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'])
    loss = F.cross_entropy
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(loss)},
                                            device=device)
    train_loss, val_loss, val_acc = [], [], []

    checkpointer = ModelCheckpoint(Path(experiment_dir) / 'checkpoints', 'checkpoint',
                                   save_interval=1, n_saved=config['epochs'])
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model})

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
        val_acc.append(avg_accuracy)
        val_loss.append(avg_loss)

        logger.info("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                     .format(engine.state.epoch, avg_accuracy, avg_loss))
        val_plotter.plot_loss_accuracy(engine, avg_accuracy, avg_loss)
        # val_plotter.plot_confusion(confusion())

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
