import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from phenotypy.misc.math import *
from PIL import Image, ImageDraw, ImageFont

try:
    import visdom
except ImportError:
    raise RuntimeError("No visdom package is found. Please install it with command: \n pip install visdom")


sns.set_style('white')
plt.rc("font", **{"sans-serif": ["Roboto"]})
pal = sns.color_palette("colorblind", 8)

pil_font = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"

__all__ = ['Plotter', 'montage_frames']


class Plotter:

    def __init__(self, out_dir, prefix='', save=True, dpi=300, formats=('.svg',), vis=False):

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=False, exist_ok=True)
        self.prefix = prefix
        self.save = save
        self.dpi = dpi
        self.formats = formats
        self.vis = None

        if vis:
            self.vis = visdom.Visdom()
            self.loss_window = self.create_plot_window('# Iterations', 'Loss', 'Loss')
            self.avg_loss_window = self.create_plot_window('# Iterations', 'Loss', 'Average loss')
            self.avg_accuracy_window = self.create_plot_window('# Iterations', 'Accuracy', 'Average accuracy')

    def savefig(self, name):

        for fmt in self.formats:

            out_file = self.out_dir / f'{self.prefix}{name}{fmt}'
            plt.savefig(out_file, dpi=self.dpi)

    def create_plot_window(self, xlabel, ylabel, title):

        return self.vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

    def plot_loss(self, engine):

        try:
            self.vis.line(X=np.array([engine.state.iteration]), Y=np.array([engine.state.output]),
                          update='append', win=self.loss_window)
        except AttributeError:
            pass

    def plot_loss_accuracy(self, engine, avg_accuracy, avg_loss):

        try:
            self.vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_accuracy]),
                          win=self.avg_accuracy_window, update='append')
            self.vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_loss]),
                          win=self.avg_loss_window, update='append')
        except AttributeError:
            pass

    def plot_confusion(self):

        pass

    def plot_activity_length_distribution(self, lengths, activity):

        fig, ax = plt.subplots()
        sns.distplot(lengths[lengths <= 100.], ax=ax, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
        self.savefig(activity)

    def plot_activity_frequency(self, count_object):

        df = pd.DataFrame({'count': count_object})
        df.index = df.index.str.replace(pat='micromovement', repl='micro.')
        df['percent'] = (df['count'] / sum(df['count'])) * 100.

        fig, ax = plt.subplots()
        ax.barh(range(len(df)), df["count"], align="center", color=pal, linewidth=1., edgecolor='k')

        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(row['count']+20, i-.1, f"{row['percent']:.1f}%", color='black')

        plt.yticks(range(len(df)), df.index)
        plt.margins(x=0.2, y=0.1)
        plt.subplots_adjust(left=0.15)
        plt.xlabel('Frequency')
        plt.ylabel('Activity')
        plt.title("Frequency of activities")
        plt.draw()

        self.savefig('activity_frequency')

    def plot_activity_length(self, df, outliers=False, unit='seconds'):

        df2 = df.copy()
        df2['activity'] = df2['activity'].str.replace(pat='micromovement', repl='micro.')

        fig, ax = plt.subplots()
        sns.boxplot(data=df2, y='activity', x=unit, showfliers=outliers,
                    palette=pal, ax=ax, linewidth=1.)

        plt.margins(x=0.2, y=0.2)
        plt.subplots_adjust(left=0.2)
        plt.xlabel(f'Length ({unit})')
        plt.ylabel('Activity')
        plt.title("Length of activities")
        plt.draw()

        self.savefig(f'activity_length_{unit}')


def montage_frames(frame_list, outfile, annot=None):

    frame_array = np.asarray(frame_list)

    if frame_array.shape[-1] != 3:
        frame_array = frame_array.transpose((1, 2, 3, 0))

    img = Image.fromarray(np.hstack(frame_array * 255.).astype('uint8'))

    if annot:
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(pil_font, 64)
        draw.text((0, 0), annot, (255, 255, 255), font=font)

    img.save(outfile)
