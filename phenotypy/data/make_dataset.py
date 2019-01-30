# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from collections import Counter
import numpy as np
import pandas as pd

import torch.utils.data as data
from phenotypy.visualization.plotting import Plotter
from phenotypy.data.loading import load_video
from phenotypy.misc.dict_tools import *


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(exists=True))
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making dataset from raw data')

    video_path = Path(input_filepath)

    for collection in ['training', 'validation', 'testing']:

        csv_file = (video_path / collection).with_suffix('.csv')
        video_list = pd.read_csv(csv_file)['video'].apply(lambda x: video_path / x).values

        plotter = Plotter(out_dir=output_filepath, formats=('.svg',), prefix=collection + '_')
        dataset = VideoCollection(video_list, output_filepath, name=collection)

        dataset.statistics(plotter)


class VideoCollection:

    def __init__(self, video_list, processed_dir, name='all'):  # TODO pass in config file for describing the experiment
        """
        VideoCollection is a lightweight wrapper than loads multiple separate video files into Video objects.
        This wrapper simply serves as a means to easily perform operations common to all videos in the collection,
        such as train-test splitting, preprocessing, and calculation of summary statistics.
        :param video_list: a list of paths to video files to be loaded into Video objects
        """
        self.video_list = video_list
        self.processed_dir = processed_dir
        self.name = name

        self.video_objects = []
        self.activity_set = set()
        self.activity_encoding = dict()
        self.annotations = []

        self._create_dataset()

    def _create_dataset(self):

        for video_path in self.video_list:

            video = Video(video_path, plotter=Plotter(self.processed_dir, prefix=video_path.stem + '_'))
            self.video_objects.append(video)

            self.annotations.extend(video.raw_annotations['activity'])
            self.activity_set = self.activity_set.union(video.activity_set)
            video.collection = self  # so each video knows which collection it belongs to

        # Generate numeric label encoding from the set of activities across all videos
        self.activity_encoding = enumerate_dict(self.activity_set)

        for video in self.video_objects:
            video.encode_labels(self.activity_encoding)

    def statistics(self, plotter):

        logging.info(f"Number of videos: {len(self.video_list)}")
        logging.info(f"Number of annotations (one or more frames): {len(self.annotations)}")
        logging.info(f"Number of unique activities: {len(self.activity_set)}")

        label_counts = Counter(self.annotations)
        plotter.plot_activity_frequency(label_counts)

        # Plot activity lengths
        raw_annotations = pd.concat([video.raw_annotations for video in self.video_objects], axis=0)
        # plotter.plot_activity_length(raw_annotations[raw_annotations['activity'] != 'rest'], unit='seconds')
        plotter.plot_activity_length(raw_annotations[raw_annotations['activity'] != 'rest'], unit='frames')

        # for video in self.video_objects:
        #     video.statistics()

    def cross_validate(self, folds=5):

        pass


class Video(data.Dataset):

    def __init__(self, video_path, data_source='MIT', plotter=None):

        # Initialise class variables
        self.video_path = video_path
        self.data_source = data_source
        self.plotter = plotter
        self.raw_annotations = pd.DataFrame()
        self.activity_set = set()
        self.activity_encoding = dict()

        self.collection = None
        self.frame_labels = None

        # Load raw video and annotations
        logging.info(f"Loading video '{video_path}'")
        self.video, self.fps = load_video(video_path)
        self._load_annotations(video_path)

    def _load_annotations(self, input_video, annotator=1):

        annotation_folder = f'Annotator_group_{annotator}' if self.data_source == 'MIT' else None
        annot_path = Path.joinpath(input_video.parent, annotation_folder, input_video.with_suffix('.txt').name)

        try:
            assert(Path.is_file(annot_path))
        except AssertionError:
            logging.warning(f"Unable to locate annotation file for video '{input_video.name}'")
            return None

        logging.info(f"Loading annotations '{annot_path}'")

        with open(annot_path) as f:
            content = f.readlines()

        self.raw_annotations = parse_mit_annotations(content) if self.data_source == 'MIT' else {}
        self.raw_annotations['seconds'] = self.raw_annotations['frames'] / self.fps
        self.activity_set = set(self.raw_annotations['activity'].unique())
        self.activity_encoding = enumerate_dict(self.activity_set)

    def encode_labels(self, encoding=None):

        if self.collection is None:
            logging.warning("Video is not part of a collection; activity labels may be incorrect")
        else:
            logging.info(f"'{self.video_path.name}' using labels from collection '{self.collection.name}'")
            if encoding != self.collection.activity_encoding:
                logging.error(f"Video activity encoding does not match its collection object f'{self.collection.name}'")
                exit(1)

        if encoding is not None:
            self.activity_encoding = encoding

        # Produce a dense list of annotations, frame-by-frame
        labels = np.zeros(shape=(self.raw_annotations.loc[len(self.raw_annotations) - 1]['end']), dtype=int)

        for _, row in self.raw_annotations.iterrows():

            label = self.activity_encoding[row['activity']]
            labels[row['start']:row['end']] = [label] * (row['end'] - row['start'])

        self.frame_labels = labels

    def statistics(self):

        print(self.raw_annotations.groupby('activity')['frames'].describe())
        self.plotter.plot_activity_length(self.raw_annotations[self.raw_annotations['activity'] != 'rest'])


def parse_mit_annotations(raw_annotations):

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
