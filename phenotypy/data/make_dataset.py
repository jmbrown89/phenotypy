# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import cv2
from collections import Counter
import numpy as np
import torch.utils.data as data
from phenotypy.visualization.plotting import *


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making dataset from raw data')

    video_path = Path(input_filepath)
    video_list = list(video_path.glob('*.mpg'))

    dataset = VideoCollection(video_list)
    # dataset.summary_statistics()


class VideoCollection:

    def __init__(self, video_list):
        """
        VideoCollection is a lightweight wrapper than loads multiple separate video files into Video objects.
        This wrapper simply serves as a means to easily perform operations common to all videos in the collection,
        such as train-test splitting, preprocessing, and calculation of summary statistics.
        :param video_list: a list of paths to video files to be loaded into Video objects
        """

        self.video_list = video_list
        self.video_objects = []

        self.behavior_set = set()
        self.behaviour_labels = dict(enumerate(self.behavior_set))

        self._create_dataset()
        self.name = 'all'  # all, training, validation, or testing

    def _create_dataset(self):

        for video_path in self.video_list:

            logging.info(f"Loading video '{video_path}'")
            video = Video(video_path)
            self.video_objects.append(video)

            self.behavior_set = self.behavior_set.union(video.unique_behaviours)
            # self.labels.extend(video.annotations[:, 1])
            video.collection = self  # so each video knows which collection it belongs to

    def summary_statistics(self):

        logging.info(f"Number of videos: {len(self.video_list)}")
        logging.info(f"Number of annotations (one or more frames): {len(self.labels)}")
        logging.info(f"Number of unique behaviours: {len(self.behavior_set)}")

        label_counts = Counter(self.labels)
        plot_label_distribution(label_counts)

    def train_test_split(self, by='video'):

        pass

    def cross_validate(self, folds=5):

        pass


class Video(data.Dataset):

    def __init__(self, video_path, data_source='MIT'):

        self.video_path = video_path
        self.data_source = data_source
        self.video = load_video(video_path)

        self.annotations = pd.DataFrame()
        self.unique_behaviours = set()
        self.load_annotations(video_path)
        self.collection = None

    def load_annotations(self, input_video, annotator=1):

        annotation_folder = f'Annotator_group_{annotator}' if self.data_source == 'MIT' else None
        annot_path = Path.joinpath(input_video.parent, annotation_folder, input_video.with_suffix('.txt').name)

        try:
            assert(Path.is_file(annot_path))
        except AssertionError:
            print(f"Unable to locate annotation file for video '{input_video.name}'")
            return None

        logging.info(f"Loading annotations '{annot_path}'")

        with open(annot_path) as f:
            content = f.readlines()

        annotations = parse_mit_annotations(content) if self.data_source == 'MIT' else {}
        self.unique_behaviours = annotations['activity'].value_counts()
        self.annotations = annotations

    def encode_labels(self):

        if self.collection is None:
            logging.error("Unable to encode labels; video must be part of a collection.")
            exit(1)

        # Produce a dense list of annotations, frame-by-frame
        labels = np.zeros(shape=(self.annotations.loc[len(self.annotations) - 1]['end'], 1))

        print(labels.shape)
        for row in self.annotations.iterrows():
            labels[row['start']:row['end']] = []


# TODO add error handling, and capacity to load various formats
def load_video(video_file):

    return cv2.VideoCapture(str(video_file))


def parse_mit_annotations(raw_annotations):

    stripped = [x.strip().split(': ')[1].split(' ') for x in raw_annotations]
    annot_data = pd.DataFrame(stripped, columns=['frames', 'activity'])

    frames = annot_data['frames'].str.split('-', expand=True)
    annot_data['start'] = pd.to_numeric(frames[0]) - 1  # index from zero...
    annot_data['end'] = pd.to_numeric(frames[1]) - 1
    del(annot_data['frames'])

    try:
        assert(len(stripped) == len(annot_data))
    except AssertionError:
        print('Duplicate entries found in annotation data. Aborting...')
        return None

    return annot_data


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
