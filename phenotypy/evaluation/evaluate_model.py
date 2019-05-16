import click
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
import yaml
from phenotypy.models.predict_model import predict
from phenotypy.misc.metrics import MultiClassEvaluator


@click.command()
@click.argument('cv_dir', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path(exists=True))
@click.argument('data_dir', type=click.Path(exists=True), required=False)
def main(cv_dir, out_dir, data_dir):
    """ Runs training based on the provided config file.
    """
    cv_dir = Path(cv_dir)
    out_dir = Path(out_dir) / cv_dir.name
    out_dir.mkdir(parents=False, exist_ok=True)
    data_dir = Path(data_dir) if data_dir else data_dir

    sub_dirs = sorted([d for d in cv_dir.glob('*/') if d.is_dir()])
    evaluator = MultiClassEvaluator(out_dir)

    for sub_dir in sub_dirs:  # nifty way to search subdirectories

        # Load model from best performing epoch
        val_results = pd.read_csv(sub_dir / 'val_results.csv', index_col=0)
        best_epoch = val_results['loss'].idxmin()
        model = (sub_dir / 'checkpoints') / f'checkpoint_model_{best_epoch + 1}.pth'

        # Load config file and get validation video
        config_file = str(next(sub_dir.glob('*.yaml')))
        config = yaml.load(open(config_file, 'rb').read(), Loader=yaml.SafeLoader)
        evaluator.encoding = config['encoding']

        if not data_dir:
            data_dir = Path.resolve(Path(config_file).parent / config['data_dir'])

        video = Path(pd.read_csv(data_dir / config['validation_csv'], index_col=0)['video'][0])
        video = data_dir / video.name

        print(f'\nCross validation: {sub_dir.name}')
        print(f'Model: {model}')
        print(f'Validation data: {video}')
        csv_out = (Path(out_dir) / video.name).with_suffix('.csv')

        try:
            print("Running...")
            results = evaluate(video, model, config_file, csv_out, config['encoding'])
            evaluator.auc(results['ground_truth'].values, results.iloc[:, 0:8].values, legend=video.name)
            evaluator.mean_accuracy(results['ground_truth'].values, results['prediction'])

            print("Done.")
        except ValueError as v:
            print("Inference failed. Skipping for now...")
            print(v)

    evaluator.run()


def evaluate(video, model, config, csv_out, encoding, stride=28):

    # Passing a float for the stride ensures it treated as a proportion of the window size
    y_true, y_pred = predict(video, model, config, stride=stride, per_frame=True,
                             save_dir=csv_out.parent, save_video=False)

    # Chop off the remaining frames, if any
    remainder = len(y_true) % stride
    if remainder > 0:
        y_true = y_true[:-remainder]
    y_pred = y_pred[:len(y_true), ...]

    df = pd.DataFrame(data=y_pred, columns=[k for k in range(0, len(encoding))])
    df['ground_truth'] = y_true
    df.to_csv(csv_out)

    df = pd.read_csv(csv_out, index_col=0)
    df['prediction'] = df.iloc[:, 0:8].idxmax(axis=1).astype(int)
    df.to_csv(csv_out)
    print(df.shape)

    # Run evaluation
    return df


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
