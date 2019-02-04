# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np


class Sampler(object):

    def __init__(self, video_objects):

        self.video_objects = video_objects
        self.idx = 0

    def sample_frames(self, v_index):

        raise NotImplementedError("Use SlidingWindowSampler or similar.")


class SlidingWindowSampler(Sampler):

    def __init__(self, video_objects, window=8, stride=1):

        Sampler.__init__(self, video_objects)
        self.window = window
        self.stride = stride
        self.attempts = 10

    def precompute_clips(self):

        clips = []
        for v_index, video in enumerate(self.video_objects):

            logging.info(f"Extracting batches from video '{video.video_path.stem}', which has {video.frames} frames")
            self.idx = 0

            sample = True
            while sample:

                for i in range(0, self.attempts):

                    sample = self.sample_frames(video)

                    if not sample:
                        break

                    idxs, labels = sample
                    if len(labels) == 1:
                        clips.append((v_index, idxs, labels[0]))
                    else:
                        continue

        return clips

    def sample_frames(self, video):

        if self.idx + self.window >= len(video.frame_labels):
            return []  # we've reached the end of the video

        annot_window = video.frame_labels[self.idx:self.idx+self.window]
        idxs = range(self.idx, self.idx + self.window)

        if len(set(annot_window)) <= 1:
            self.idx += self.stride
        else:
            self.idx += self.window

        return idxs, list(set(annot_window))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # main()




