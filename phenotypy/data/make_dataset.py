# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from collections import Counter
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from phenotypy.data.loading import load_video
from phenotypy.data.sampling import SlidingWindowSampler
from phenotypy.data.transforms import *
from phenotypy.misc.dict_tools import *
from phenotypy.misc.utils import parse_config
from phenotypy.visualization.plotting import *

logger = logging.getLogger(__name__)


@click.command()
@click.argument('config', type=click.Path(exists=True))
def main(config):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('Making dataset from raw data')

    config = parse_config(config)
    # training_data = dataset_from_config(config, name='training')
    # for i, (clip, target) in enumerate(training_data):
    #     montage_frames(clip, training_data.label_encoding[target])

    train_loader, val_loader = create_data_loaders(config)
    train_loader.dataset.statistics(plotter=Plotter('../../data/processed', '_'))

    # for inputs, targets in train_loader:
    #
    #     for clip, target in zip(inputs, targets.numpy()):
    #         montage_frames(clip, train_loader.dataset.label_encoding[target])


def create_data_loaders(config):

    training_data = dataset_from_config(config, name='training')
    validation_data = dataset_from_config(config, name='validation')

    train_loader = data.DataLoader(training_data,
                                   batch_size=config['batch_size'],
                                   shuffle=True,
                                   num_workers=1,  # TODO check to ensure same video not accessed multiple times
                                   pin_memory=False)  # TODO this may be problematic

    validation_loader = data.DataLoader(validation_data,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1,
                                        pin_memory=False)

    return train_loader, validation_loader


def dataset_from_config(config, name='training'):

    data_dir, save_dir = Path(config['data_dir']), Path(config['out_dir'])
    csv_file = config[f'{name}_csv']

    video_list = pd.read_csv(data_dir / csv_file)['video'].apply(lambda x: data_dir / x).values
    logger.info(f"Found {len(video_list)} videos in '{data_dir}'")
    loader = VideoCollection(video_list, save_dir, config, name=name)
    return loader


class VideoCollection(data.Dataset):

    def __init__(self, video_list, processed_dir, config, name='all'):
        """
        VideoCollection is a lightweight wrapper than loads multiple separate video files into Video objects.
        This wrapper serves to generate frame sequences from raw footage, without breaking them down into frames.
        :param video_list: a list of paths to video files to be loaded into Video objects
        """
        self.video_list = video_list
        self.processed_dir = processed_dir
        self.config = config
        self.name = name
        self.debug = config.get('debug', False)
        self.limit_clips = None if not self.debug else config.get('limit_clips', None)

        self.video_objects = []
        self.activity_set = set()
        self.activity_encoding = dict()
        self.label_encoding = dict()
        self.annotations = []

        self._create_dataset()
        self._preprocessing()

    def _create_dataset(self):
        """
        Create video objects for each video in the collection, and create a global activity set and encoding. Once
        created, the video annotations will be encoded accordingly.
        """

        for video_path in self.video_list:

            video = Video(video_path, plotter=Plotter(self.processed_dir, prefix=video_path.stem + '_'))
            self.video_objects.append(video)

            self.annotations.extend(video.raw_annotations['activity'])
            self.activity_set = self.activity_set.union(video.activity_set)
            video.collection = self  # so each video knows which collection it belongs to

        # Generate numeric label encoding from the set of activities across all videos
        # TODO - make this more flexible. Might want to only annotate certain behaviours
        self.activity_encoding = enumerate_dict(tuple(sorted(list(self.activity_set))))
        self.label_encoding = reverse_dict(self.activity_encoding)
        self.no_classes = len(self.activity_encoding)

        for video in self.video_objects:
            video.encode_labels(self.activity_encoding)

        # Precompute batches with which to train
        window, stride = self.config['clip_length'], self.config['clip_stride']
        self.sampler = SlidingWindowSampler(self.video_objects, window=window, stride=stride, limit_clips=self.limit_clips)
        self.clips = self.sampler.precompute_clips()
        self.height, self.width = self.video_objects[0].height, self.video_objects[0].width
        logger.info(f'Clips extracted for {self.name}: {len(self.clips)}')

    def _preprocessing(self):

        transforms = [ToPIL(), Scale((128, 128))]

        if self.name == 'training':
            transforms.append(RandomHorizontalFlip())  # TODO add more augmentation with config options

        transforms.append(ToTensor())
        self.spatial_transform = Compose(transforms)

    def __len__(self):
        """
        :return: number of videos in the collection
        """

        return len(self.clips)

    def __getitem__(self, item):
        """
        Returns a single clip, parametrised by this collection's sampler object.
        :param item: index of the batch to be generated
        :return: a 4-dimensional Torch tensor of sampled video frames (channels x frames x rows x cols)
        """

        v_idx, f_idxs, label = self.clips[item]
        clip = self.video_objects[v_idx].get_frames(f_idxs)
        self.spatial_transform.randomize_parameters()  # once per clip!

        try:
            clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        except TypeError:
            logger.critical(f"Unable to load clip at location {f_idxs[0]} - {f_idxs[-1]} "
                            f"from video '{self.video_objects[v_idx].video_path.name}'")
            exit(1)

        return clip, label

    def statistics(self, plotter):
        """
        Calculate statistics over the whole collection, such as number of videos, frequency of activities, and length of
        time spent doing each activity.
        :param plotter: plotting object with which to generate plots
        """

        sample_stats = Counter([sample[-1] for sample in self.clips])
        logger.info(f"Number of videos: {len(self.video_list)}")
        logger.info(f"Number of annotations (one or more frames): {len(self.annotations)}")
        logger.info(f"Number of unique activities: {self.no_classes}")
        logger.info(f"Label distribution of sampled data: {sample_stats}")

        # Plot activity lengths and frequency of annotations
        raw_annotations = pd.concat([video.raw_annotations for video in self.video_objects], axis=0)
        label_counts = Counter(self.annotations)
        plotter.plot_activity_length(raw_annotations[raw_annotations['activity'] != 'rest'], unit='frames')
        plotter.plot_activity_frequency(label_counts)


class Video:

    def __init__(self, video_path, data_source='MIT', plotter=None):
        """
        This class is wrapper around individual video clips, and serves to load frame sequences and associated
         annotations for training.
        :param video_path: the file path of the video to load
        :param data_source: which dataset the video comes from (e.g. MIT)
        :param plotter: an optional object which will do all the plotting associated with this video
        """

        # Initialise class variables
        self.video_path = video_path
        self.data_source = data_source
        self.plotter = plotter

        self.raw_annotations = pd.DataFrame()
        self.activity_set = set()
        self.activity_encoding = dict() # str --> numeric
        self.label_encoding = dict()  # numeric --> str

        self.collection = None
        self.frame_labels = None

        # Load raw video and annotations
        logger.info(f"Loading video '{video_path}'")
        self.video, self.fps, self.channels, self.frames, self.height, self.width = load_video(video_path)
        self._load_annotations(video_path)

    def _load_annotations(self, input_video, annotator=1):
        """
        This function will load the annotations that correspond with a particular video. It's not very dataset-agnostic
        at the moment, as it only works for the MIT dataset. Once the annotations are loaded, it will choose an encoding
        (activity text --> numeric) which can be reset by the VideoCollections class.
        :param input_video: file path for video
        :param annotator: the MIT annotator number to use (1 or 2)
        """

        annotation_folder = f'Annotator_group_{annotator}' if self.data_source == 'MIT' else None
        annot_path = Path.joinpath(input_video.parent, annotation_folder, input_video.with_suffix('.txt').name)

        try:
            assert(Path.is_file(annot_path))
        except AssertionError:
            logger.warning(f"Unable to locate annotation file for video '{input_video.name}'")
            exit(1)

        # logging.info(f"Loading annotations '{annot_path}'")

        with open(annot_path) as f:
            content = f.readlines()

        self.raw_annotations = parse_mit_annotations(content) if self.data_source == 'MIT' else {}
        self.raw_annotations['seconds'] = self.raw_annotations['frames'] / self.fps
        self.activity_set = set(self.raw_annotations['activity'].unique())
        self.activity_encoding = enumerate_dict(self.activity_set)
        self.label_encoding = reverse_dict(self.activity_encoding)

    def get_frames(self, idxs):
        """
        This function will simply extract frames at the indices specified, in that order, and return them as a list.
        This function does not return the corresponding annotations, as the indices are assumed to be provided in such a
        way that the label would be the same for all frames.
        :param idxs: indices of the video from which frames should be sampled
        :return: sampled frames
        """

        frames = []
        for idx in idxs:

            self.video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            res, frame = self.video.read()  # BGR!!!!
            attempt = 0

            while not res and attempt < 5:
                attempt += 1
                logger.warning(f"Failed attempt #{attempt} reading '{self.video_path.name}'"
                               f" (opened = {self.video.isOpened()})")
                self.video.release()
                self.video, self.fps, self.channels, self.frames, self.height, self.width = load_video(self.video_path)
                self.video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                res, frame = self.video.read()

            if not res:
                return None

            # TODO make this configurable
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.pad(frame, ((8, 8), (0, 0), (0, 0)), mode='symmetric')

            crop = (frame.shape[1] - frame.shape[0]) // 2
            frame = frame[:, crop:-crop, :]
            frames.append(frame)

        return frames

    def encode_labels(self, encoding=None):
        """
        This function will convert the textual annotations (MIT format) into per-frame annotations, and subsequently
        apply the encoding (activity text --> numeric --> one-hot) specified. If the video is part of a collection,
        the specified encoding must match that of the collection to which it belongs.
        :param encoding:
        """

        if self.collection is None:
            logger.error("Video must be part of a VideoCollection object in order to encode labels.")
        else:
            # logging.info(f"'{self.video_path.name}' using labels from collection '{self.collection.name}'")
            if encoding != self.collection.activity_encoding:
                logger.error(f"Video activity encoding does not match its collection object f'{self.collection.name}'")
                exit(1)

        if encoding is not None:
            self.activity_encoding = encoding
            self.label_encoding = reverse_dict(encoding)

        # Produce a dense list of annotations, frame-by-frame
        labels = np.zeros(shape=(self.raw_annotations.loc[len(self.raw_annotations) - 1]['end']), dtype=int)

        for _, row in self.raw_annotations.iterrows():

            label = self.activity_encoding[row['activity']]
            labels[row['start']:row['end']] = [label] * (row['end'] - row['start'])

        self.frame_labels = labels

    def statistics(self):
        """
        Calculate video-specific statistics of the annotations.
        """

        print(self.raw_annotations.groupby('activity')['frames'].describe())
        self.plotter.plot_activity_length(self.raw_annotations[self.raw_annotations['activity'] != 'rest'])


def parse_mit_annotations(raw_annotations):
    """
    Convert and load the raw MIT activity annotations from text files into a structured DataFrame.
    :param raw_annotations: the raw annotations from text files
    :return: the parsed annotations
    """

    stripped = [x.strip().split(': ')[1].split(' ') for x in raw_annotations]
    raw_data = pd.DataFrame(stripped, columns=['frame_window', 'activity'])

    frames = raw_data['frame_window'].str.split('-', expand=True)
    raw_data['start'] = pd.to_numeric(frames[0]) - 1  # index from zero...
    raw_data['end'] = pd.to_numeric(frames[1]) - 1
    raw_data['frames'] = raw_data['end'] - raw_data['start']

    del(raw_data['frame_window'])

    try:
        assert(len(stripped) == len(raw_data))
    except AssertionError:
        print('Duplicate entries found in annotation data. Aborting...')
        return None

    return raw_data


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
