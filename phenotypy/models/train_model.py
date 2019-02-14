import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
from torch import optim
import torch.utils.data as data
import torch.nn.functional as F
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from phenotypy.data.make_dataset import parse_config, create_data_loaders
from phenotypy.models import resnet
from phenotypy.visualization.plotting import Plotter


@click.command()
@click.argument('config', type=click.Path(exists=True))
def main(config):
    """ Runs training based on the provided config file.
    """
    logger = logging.getLogger(__name__)
    logger.info('Training started')

    config_dict = parse_config(Path(config))
    train(**config_dict)


def train(**config):

    train_loader, val_loader = create_data_loaders(**config)
    train_plotter = Plotter(config['out_dir'], vis=config['visdom'])
    val_plotter = Plotter(config['out_dir'], vis=config['visdom'])

    logging.info("Initialising model")
    model = resnet.resnet10(  # TODO https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md#shortcuttype
        num_classes=config['no_classes'],
        sample_size=config['transform']['scale'],
        sample_duration=config['clip_length'])

    params = resnet.get_fine_tuning_parameters(model)
    optimizer = optim.Adam(params)
    loss = F.cross_entropy
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'loss': Loss(loss)},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % config['log_interval'] == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
            train_plotter.plot_loss(engine)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_loss))
        train_plotter.plot_loss_accuracy(engine, avg_accuracy, avg_loss)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_loss))
        val_plotter.plot_loss_accuracy(engine, avg_accuracy, avg_loss)

    trainer.run(train_loader, max_epochs=config['epochs'])


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
