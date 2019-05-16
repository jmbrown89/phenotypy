# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import cv2
from phenotypy.data.make_dataset import Video
from skvideo.io import FFmpegWriter

font = cv2.FONT_HERSHEY_COMPLEX
annot_pos = (10, 25)
font_scale = 0.7
true_color = (255, 0, 255)
pred_color = (255, 255, 0)
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


def play_video(video_path, activity_encoding=None, predicted_labels=None, save_video=None):
    """
    Plays a video from file with annotations overlaid. Play/pause with 'p' and quit with 'q', if save_video=None.
    :param video_path: path to video file to be played
    :param activity_encoding: dictionary mapping integer labels to activities as strings
    :param predicted_labels: prediction annotations to overlay (in addition to the true labels)
    :param save_video: optional file path to which the annotated video can be saved
    """

    video_path = Path(video_path)
    video_obj = Video(video_path)
    video_obj.encode_labels(activity_encoding)

    video = video_obj.video
    pause = int(video_obj.fps / 2.)
    true_labels = video_obj.frame_labels
    activities = video_obj.label_encoding
    collection = video_obj.collection.name if video_obj.collection else 'no source'
    index = 0

    writer = None
    if save_video:
        writer = FFmpegWriter(save_video)

    while video.isOpened():

        ret, frame = video.read()
        key = cv2.waitKey(1) & 0xff

        if not ret:
            break

        index += 1

        if cv2.waitKey(pause) & key == ord('p') and not save_video:

            while True:

                key2 = cv2.waitKey(1) or 0xff
                if key2 == ord('p'):
                    break

        annotate_frame(frame, activities[true_labels[index]])
        if predicted_labels is not None:

            try:
                annotate_frame(frame, activities[predicted_labels[index]], 20)
            except IndexError:
                break

        cv2.imshow(f'{video_path.name} ({collection})', frame)

        if save_video:
            writer.writeFrame(frame[..., ::-1])

        # Press Q on keyboard to  exit
        if cv2.waitKey(pause) & 0xFF == ord('q'):
            break

    video.release()
    if save_video:
        writer.close()
    cv2.destroyAllWindows()


def annotate_frame(f, text, y_offset=0):
    """
    Helper function for displaying a frame and its annotation
    :param f: the frame as a numpy array
    :param text: a string to overlay on the frame
    :param y_offset: the amount to shift the text overlay in the y-direction
    """

    cv2.putText(img=f, text=text, org=(annot_pos[0], annot_pos[1] + y_offset),
                fontFace=font, fontScale=.75, color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=3)
    cv2.putText(img=f, text=text, org=(annot_pos[0], annot_pos[1] + y_offset),
                fontFace=font, fontScale=.75, color=pred_color if y_offset else true_color,
                lineType=cv2.LINE_AA,
                thickness=2)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


