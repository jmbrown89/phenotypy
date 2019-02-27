import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import torch
from torch.nn import Softmax
from torch.utils import data
from tqdm import tqdm
import numpy as np
from phenotypy.data.make_dataset import VideoCollection, parse_config
from phenotypy.visualization.visualize import play_video




@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('config_path', type=click.Path(exists=True))
def main(video_path, config_path):
    """ Runs training based on the provided config file.
    """
    predict(video_path, config_path)


def predict(video_path, config_path, stride=1.0, device='cuda'):

    config = parse_config(Path(config_path))
    config['clip_length'] = 16
    save_dir = Path(config['out_dir']) / Path(video_path).stem
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset = VideoCollection([video_path], save_dir, config, name='testing')

    dataset.sample_clips(stride=stride, testing=True)  # the latter is very important - ensures we get a mix of labels
    loader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    model = torch.load('/home/user/Research/Projects/HCA/Results/default/checkpoints/checkpoint_model_1.pth').eval()
    model.to(device)

    outputs = []
    true_frames = []

    for x, y in tqdm(loader):
        outputs.append(model(x.to(device)).detach().to('cpu').numpy())
        true_frames.extend(list(np.hstack(y.numpy())))

    y_pred = np.argmax(np.vstack(outputs), axis=1)
    pred_frames = window_label_to_frames(y_pred, size=config['clip_length'])
    pd.DataFrame(pd.Series(y_pred, name='label')).to_csv(save_dir / 'prediction.csv')
    play_video(video_path, dataset.activity_encoding, predicted_labels=pred_frames, save_video=save_dir / 'prediction.mp4')


def window_label_to_frames(arr, size=16):

    frame_labels = []
    for label in arr:
        frame_labels.extend([label] * size)

    return frame_labels


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
