from pathlib import Path
from phenotypy.data.make_dataset import load_single
import torch
from torch import nn
import click
from tqdm import tqdm
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('Dark2')
from scipy.stats import mode


@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('save_dir', type=click.Path(exists=True), default=None)
def umap_analysis(video_path, model_path, config_path, save_dir, device='cuda'):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=False, exist_ok=True)
    video_path = Path(video_path)
    loader, config = load_single(str(video_path), config_path, testing=True, stride=1., batch_size=8)

    np_file = (save_dir / video_path.name).with_suffix('.npz')
    features, labels = [], []

    if not np_file.exists():

        # Load the model for feature extraction
        model = torch.load(model_path)
        model = nn.Sequential(*list(model.children())[:-1]).eval()
        for param in model.parameters():
            param.requires_grad = False

        model.to(device)

        for x, y in tqdm(loader):  # TODO make this smarter so that we can restart incomplete runs

            if device == 'cuda':
                x = x.to(device)
                output = model(x).detach().to('cpu')
            else:
                output = model(x).detach()

            features.append(output.squeeze().numpy())

            y, _ = mode(y, axis=1)
            labels.append(y)

        X = np.concatenate(features, axis=0)
        y = np.concatenate(labels, axis=0).squeeze()
        np.savez(np_file, X=X, y=y)

    else:
        files = np.load(np_file)
        X, y = files['X'], files['y']

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)
    legend = loader.dataset.label_encoding

    for c in np.unique(y):

        idx = y == c
        plt.scatter(embedding[idx, 0], embedding[idx, 1], c=sns.color_palette()[c],
                    edgecolors='none', s=8, alpha=1.0, label=legend[c].capitalize())

    plt.title('UMAP projection of HCA video')
    plt.legend()
    plt.savefig((save_dir / video_path.name).with_suffix('.png'), dpi=300)


if __name__ == '__main__':

    umap_analysis()
