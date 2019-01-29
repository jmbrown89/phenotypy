import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.rc("font", **{"sans-serif": ["Roboto"]})
# plt.rc('font', family='Ubuntu')


def plot_label_distribution(count_object):

    df = pd.DataFrame({'count': count_object})
    df.index = df.index.str.replace(pat='micromovement', repl='micro.')
    df['percent'] = (df['count'] / sum(df['count'])) * 100.

    fig, ax = plt.subplots()

    ax.barh(range(len(df)), df["count"], align="center",
            color=sns.color_palette("hls", len(df)))

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['count'], i-.1, f"{row['percent']:.1f}%", color='black')

    plt.yticks(range(len(df)), df.index)
    plt.margins(x=0.2, y=0.1)
    plt.subplots_adjust(left=0.15)
    plt.xlabel('Frequency')
    plt.ylabel('Behaviour')

    plt.title("Frequency of behaviours")
    plt.show()
