import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from collections import Counter
from phenotypy.data.make_dataset import load_single
from phenotypy.visualization.visualize import play_video
from phenotypy.visualization.plotting import Plotter

@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path(exists=True), default=None)
@click.argument('device', type=str, default='cuda')
def main(video_path, model_path, config_path, out_dir, device):
    """ Runs training based on the provided config file.
    """
    predict(video_path, model_path, config_path, device=device, save_dir=out_dir, save_video=out_dir is not None)


def predict(video_path, model_path, config_path, stride=7, device='cuda', per_frame=True, save_dir='./', save_video=False):

    # Create a loader for a single video
    video_path = Path(video_path)
    save_dir = Path(save_dir)
    print(f"Loading video '{video_path.stem}'")
    loader, config = load_single(str(video_path), config_path, testing=True, stride=stride, batch_size=1)

    # Create plotter object to plot... stuff
    # plotter = Plotter(Path(save_dir) / 'plots', prefix=video_path.with_suffix('').name + '_')
    # df = loader.dataset.video_objects[0].raw_annotations
    # # plotter.plot_activity_length(df, outliers=False)
    # plotter.plot_activity_frequency(Counter(df['activity']))
    np_file = (save_dir / video_path.name).with_suffix('.npz')

    if not np_file.exists():

        # Load the model for evaluation
        model = torch.load(model_path).eval()
        model.to(device)
        y_preda, y_true = [], []

        for x, y in tqdm(loader):  # TODO make this smarter so that we can restart incomplete runs

            if device == 'cuda':
                x = x.to(device)
                output = model(x).detach().to('cpu')
            else:
                output = model(x).detach()

            y_preda.append(torch.softmax(output, 1).numpy())
            y_true.extend(list(np.hstack(y.numpy())))

        y_preda = np.vstack(y_preda)
        y_true = np.asarray(y_true)
        np.savez(np_file, y_preda=y_preda, y_true=y_true)

    else:
        files = np.load(np_file)
        y_preda, y_true = files['y_preda'], files['y_true']

    if per_frame:

        clip_size = config['clip_length']

        if not 0 < stride <= clip_size or not isinstance(stride, int):
            raise ValueError(
                f'For clip level annotations, stride must be a non-zero positive integer'
                f'less than or equal to the clip length ({clip_size})')

        if stride == clip_size:
            y_pred, y_preda = unstrided_labels_to_frames(y_preda, clip_size=clip_size)  # TODO check this works
        else:
            y_pred, y_preda = strided_labels_to_frames(y_preda, clip_size=clip_size, stride=stride)
            # y_true = np.hstack(loader.dataset.clips.loc[::4]['label'])

        if save_video:
            print("Annotating video")
            save_dir = Path(save_dir)
            pd.DataFrame(pd.Series(y_pred, name='y_pred')).to_csv(save_dir / video_path.with_suffix('.csv').name)
            play_video(video_path, loader.dataset.activity_encoding, predicted_labels=y_pred,
                       save_video=save_dir / video_path.with_suffix('.mp4').name)

    return y_true, y_preda


def unstrided_labels_to_frames(clip_preds, clip_size=28):
    """
    Converts clip-level predictions to frame level annotations, assuming no stride. One can use strided_label_to_frames
    to achieve the same result where clip_size == stride, but it's computationally more expensive.
    :param clip_preds: clip-level predictions of shape N x C. N is the number of clips, and c is the number of classes
    :param clip_size: the number of frames per clip, used as a multiplier for the clip-levle annotations
    :return: an array containing the frame-level annotations
    """

    clip_labels = np.argmax(clip_preds, axis=1)

    frame_probs, frame_labels = [], []
    for prob, label in zip(clip_preds, clip_labels):  # annoying that one can't use a list comprehension for this
        frame_probs.append(np.tile(prob, (clip_size, 1)))
        frame_labels.extend([label] * clip_size)

    return np.asarray(frame_labels), np.concatenate(frame_probs, axis=0)


def strided_labels_to_frames(clip_preds, clip_size=28, stride=1):

    # We first duplicate each clip, and create a dictionary to store overlapping predictions
    frame_preds = np.repeat(clip_preds, repeats=clip_size, axis=0)
    overlapping_preds = defaultdict(list)

    clip_pos = 0
    for window in range(0, frame_preds.shape[0], clip_size):

        for window_pos, prob in enumerate(frame_preds[window:window + clip_size]):
            overlapping_preds[clip_pos + window_pos].append(prob)
        clip_pos += stride

    no_frames = len(overlapping_preds.keys())
    frame_labels = np.zeros(shape=(no_frames,), dtype=int)
    frame_probs = np.zeros(shape=(no_frames, clip_preds.shape[1]))

    for frame, preds in overlapping_preds.items():
        label = np.stack(preds, -1).max(1).argmax()
        mean_prob = np.mean(np.stack(preds, 0), axis=0)
        frame_labels[frame] = label
        frame_probs[frame] = mean_prob

    return frame_labels, frame_probs


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
