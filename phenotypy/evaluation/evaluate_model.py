import click
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from phenotypy.models.predict_model import predict
from phenotypy.misc.metrics import mean_accuracy


@click.command()
@click.argument('video', type=click.Path(exists=True))
@click.argument('model', type=click.Path(exists=True))
@click.argument('config', type=click.Path(exists=True))
def main(video, model, config):
    """ Runs training based on the provided config file.
    """
    evaluate(video, model, config)


def evaluate_cross_validation():
    pass


def evaluate(video, model, config):

    # Passing a float for the stride ensures it treated as a proportion of the window size
    y_true, y_pred = predict(video, model, config, stride=1.0, device='cpu', per_frame=False, save_dir=None)
    print(mean_accuracy(y_true, y_pred))


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
