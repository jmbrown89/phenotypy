import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from phenotypy.misc.math import *


sns.set_style('white')
plt.rc("font", **{"sans-serif": ["Roboto"]})
pal = sns.color_palette("colorblind", 8)


class Plotter:

    def __init__(self, out_dir, prefix='', save=True, dpi=300, formats=('.svg',)):

        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.save = save
        self.dpi = dpi
        self.formats = formats

    def savefig(self, name):

        for fmt in self.formats:

            out_file = self.out_dir / f'{self.prefix}{name}{fmt}'
            plt.savefig(out_file, dpi=self.dpi)

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


def montage_frames(frame_list, title):

    plt.imshow(np.hstack(frame_list))
    plt.title(title)
    plt.draw()
