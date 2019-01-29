
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import cv2
from phenotypy.data.make_dataset import Video

font = cv2.FONT_HERSHEY_SIMPLEX
annot_pos = (10, 25)
font_scale = 0.7
font_color = (255, 0, 0)
line_type = 2


@click.command()
@click.argument('video_path', type=click.Path(exists=True))
def main(video_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Visualising video footage')

    video_path = Path(video_path)
    play_video(video_path)


def play_video(video_path):

    video_obj = Video(video_path)
    video = video_obj.video
    annotations = video_obj.annotations
    collection = video_obj.collection.name if video_obj.collection else 'no source'
    frame_index = 0
    annot_index = 0

    print(annotations)

    while video.isOpened():

        ret, frame = video.read()

        if ret:

            #annot_text = annotations[frame_index]

             # annot_range = None


            cv2.putText(frame, 'Hello World!', annot_pos, font, font_scale, font_color, line_type)
            cv2.imshow(f'{video_path.name} ({collection})', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


