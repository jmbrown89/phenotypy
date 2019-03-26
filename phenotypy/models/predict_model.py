import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from phenotypy.data.make_dataset import load_single
from phenotypy.visualization.visualize import play_video


@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('device', type=str, default='cuda')
def main(video_path, model_path, config_path, device):
    """ Runs training based on the provided config file.
    """
    predict(video_path, model_path, config_path, device=device)


def predict(video_path, model_path, config_path, stride=7, device='cuda', per_frame=True, save_dir='./'):

    # Create a loader for a single video
    print(f"Loading video '{Path(video_path).stem}'")
    loader, config = load_single(video_path, config_path, testing=per_frame, stride=stride, batch_size=2)

    # Load the model for evaluation
    model = torch.load(model_path).eval()
    model.to(device)

    dev_str = 'CPU' if device == 'cpu' else 'GPU'
    print(f"Running inference on {dev_str}")
    y_preda, y_true = [], []

    for x, y in tqdm(loader):  # TODO make this smarter so that we can restart incomplete runs

        if device == 'cuda':
            x = x.to(device)
            output = model(x).detach().to('cpu')
        else:
            output = model(x).detach()

        y_preda.append(output.numpy())
        y_true.extend(list(np.hstack(y.numpy())))

    y_pred = np.vstack(y_preda)
    y_true = np.asarray(y_true)

    if per_frame:

        clip_size = config['clip_length']

        if not 0 < stride <= clip_size or not isinstance(stride, int):
            raise ValueError(
                f'For clip level annotations, stride must be a non-zero positive integer'
                f'less than or equal to the clip length ({clip_size})')

        if stride == clip_size:
            y_pred = unstrided_labels_to_frames(y_pred, clip_size=clip_size)
        else:
            y_pred = strided_labels_to_frames(y_pred, clip_size=clip_size, stride=stride)

        if save_dir:
            print("Annotating video")
            save_dir = Path(save_dir)
            pd.DataFrame(pd.Series(y_pred, name='label')).to_csv(save_dir / 'prediction.csv')
            play_video(video_path, loader.dataset.activity_encoding, predicted_labels=y_pred,
                       save_video=save_dir / 'prediction.mp4')

    return y_true, y_pred


def unstrided_labels_to_frames(clip_preds, clip_size=28):
    """
    Converts clip-level predictions to frame level annotations, assuming no stride. One can use strided_label_to_frames
    to achieve the same result where clip_size == stride, but it's computationally more expensive.
    :param clip_preds: clip-level predictions of shape N x C. N is the number of clips, and c is the number of classes
    :param clip_size: the number of frames per clip, used as a multiplier for the clip-levle annotations
    :return: an array containing the frame-level annotations
    """

    clip_preds = np.argmax(clip_preds, axis=1)

    frame_labels = []
    for label in clip_preds:  # annoying that one can't use a list comprehension for this
        frame_labels.extend([label] * clip_size)

    return np.asarray(frame_labels)


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

    for frame, preds in overlapping_preds.items():
        label = np.stack(preds, -1).max(1).argmax()
        frame_labels[frame] = label

    return np.asarray(frame_labels)

if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
