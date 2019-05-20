from pathlib import Path
import cv2
from phenotypy.misc.utils import parse_config
import torch.utils.data as data


(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')


# TODO add error handling, and capacity to manage various formats
def load_video(video_file):
    """
    Creates a cv2 VideoCapture object, and loads its associated metadata.
    :param video_file:
    :return: video object, frames per second, frame width, frame height and number of frames.
    """

    if not video_file.exists():
        raise FileNotFoundError(f"Video file '{video_file}' does not exist!")

    video = cv2.VideoCapture(str(video_file))
    fps = video.get(cv2.CV_CAP_PROP_FPS) if int(major_ver) < 3 else video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CV_CAP_PROP_FRAME_WIDTH)) if int(major_ver) < 3 else int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)) if int(major_ver) < 3 else int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(video.get(cv2.CV_CAP_PROP_FRAME_COUNT)) if int(major_ver) < 3 else int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame = None
    attempts = 0
    while not frame and attempts < 10:
        res, frame = video.read()  # BGR!!!!
        attempts += 1

    try:
        channels = frame.shape[-1]
    except AttributeError:
        raise IOError(f"Unable to load frame from video {video_file}")

    return video, fps, channels, frames, height, width  #
