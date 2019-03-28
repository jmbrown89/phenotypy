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
        self.logger = logging.getLogger(__name__)

    def precompute_clips(self):

        raise NotImplementedError("Use SlidingWindowSampler or similar.")

    def sample_frames(self, v_index):

        raise NotImplementedError("Use SlidingWindowSampler or similar.")


class SlidingWindowSampler(Sampler):

    def __init__(self, video_objects, window=8, stride=1.0, limit_clips=None):

        Sampler.__init__(self, video_objects)
        self.window = window
        self.stride = stride if isinstance(stride, int) else int(round(window * stride))
        self.attempts = 10
        self.limit_clips = limit_clips
        self.testing = False

    def precompute_clips(self, testing=False):

        clips = []
        self.testing = testing

        for v_index, video in enumerate(self.video_objects):

            self.idx = 0

            sample = True
            while sample:

                for i in range(0, self.attempts):

                    sample = self.sample_frames(video)

                    if not sample:
                        break

                    idxs, labels = sample
                    unique = list(set(labels))

                    if len(unique) == 1 or testing:
                        annot = labels if testing else labels[0]
                        clips.append((v_index, idxs[0], idxs[-1], annot))
                        break  # no need for more attempts
                    else:
                        continue

                if self.limit_clips and len(clips) >= self.limit_clips:
                    return pd.DataFrame(clips, columns=['vidx', 'start', 'end', 'label'])

        return pd.DataFrame(clips, columns=['vidx', 'start', 'end', 'label'])

    def sample_frames(self, video):

        if self.idx + self.window >= len(video.frame_labels):
            return []  # we've reached the end of the video

        annot_window = video.frame_labels[self.idx:self.idx+self.window]
        idxs = range(self.idx, self.idx + self.window)

        if len(set(annot_window)) == 1 or self.testing:
            self.idx += self.stride
        else:
            self.idx += self.window

        return idxs, annot_window


class SlidingWindowOversampler(SlidingWindowSampler):

    def __init__(self, video_objects, window=8, stride=1.0, limit_clips=None):

        SlidingWindowSampler.__init__(self, video_objects, window, stride, limit_clips)

    def precompute_clips(self, testing=False):

        # Sample clips in the same way as
        clips = super().precompute_clips(testing)

        if testing:
            return clips

        # Oversample the clips
        oversampled = []
        majority_n = max([g.shape[0] for _, g in clips.groupby('label')])
        for label, clip_group in clips.groupby('label'):
            ratio = majority_n / clip_group.shape[0]
            oversampled.append(clip_group.sample(frac=ratio, replace=True))

        return pd.concat(oversampled)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # main()




