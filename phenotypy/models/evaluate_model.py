import click
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

from sklearn.metrics import confusion_matrix
import pandas as pd
import torch

@click.command()
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('experiment_name', type=str)
@click.argument('eval_data', type=click.Path(exists=True))
def main(model_dir, experiment_name, eval_data):
    """ Runs training based on the provided config file.
    """

    model_dir = Path(model_dir)
    result = pd.read_csv(model_dir / 'val_results.csv')

    epoch = 1
    best_model = (model_dir / 'checkpoints' / experiment_name).with_suffix(f'_model_{epoch}.pth')

    loader = None
    evaluate(best_model, loader)


def evaluate(model_path, data_loader):

    model = torch.load(model_path)






def confusion(y_true, y_pred, labels):

    mat = confusion_matrix(y_true, y_pred)
    pd.DataFrame(mat, columns=labels, index=labels)
    print(mat)


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
