# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import cv2
from phenotypy.data.make_dataset import Video
from phenotypy.misc.dict_tools import *

font = cv2.FONT_HERSHEY_SIMPLEX
annot_pos = (10, 25)
font_scale = 0.7
font_color = (255, 0, 255)
line_type = 2


@click.command()
@click.argument('video_path', type=click.Path(exists=True))
def main(video_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Playing video')

    video_path = Path(video_path)
    play_video(video_path)


def play_video(video_path):
    """
    Plays a video from file with annotations overlaid. Play/pause with 'p' and quit with 'q' (x button does not work).
    :param video_path: path to video file to be played
    """

    video_obj = Video(video_path)
    video_obj.encode_labels()

    video = video_obj.video
    pause = int(video_obj.fps / 2.)
    labels = video_obj.frame_labels
    activities = reverse_dict(video_obj.activity_encoding)
    collection = video_obj.collection.name if video_obj.collection else 'no source'
    index = 0

    def annotate_frame(frame, annot_text):
        """
        Helper function for displaying a frame and its annotation
        :param frame: the frame as a numpy array
        :param annot_text: a string to overlay on the frame
        """
        cv2.putText(frame, annot_text, annot_pos, font, font_scale, font_color, line_type)
        cv2.imshow(f'{video_path.name} ({collection})', frame)

    while video.isOpened():

        ret, frame = video.read()
        key = cv2.waitKey(1) & 0xff

        if not ret:
            break

        annot_text = activities[labels[index]]
        index += 1

        if cv2.waitKey(pause) & key == ord('p'):

            while True:

                key2 = cv2.waitKey(1) or 0xff
                annotate_frame(frame, annot_text)

                if key2 == ord('p'):
                    break

        annotate_frame(frame, annot_text)

        # Press Q on keyboard to  exit
        if cv2.waitKey(pause) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


