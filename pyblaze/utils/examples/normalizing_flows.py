# pylint: disable=missing-docstring
import torch
import torch.optim as optim
from torch.utils.data.dataset import TensorDataset
import matplotlib.pyplot as plt
import pyblaze.nn as xnn
import pyblaze.plot as P

def train(engine, data, lr=1e-2, epochs=1000):
    data_ = torch.as_tensor(data, dtype=torch.float)
    # pylint: disable=no-member
    loader = TensorDataset(data_).loader(batch_size=4096)

    optimizer = optim.Adam(engine.model.parameters(), lr=lr)

    return engine.train(
        loader,
        epochs=epochs,
        optimizer=optimizer,
        loss=xnn.TransformedNormalLoss(),
        callbacks=[
            xnn.EpochProgressLogger(),
        ],
        gpu=False
    )

def train_and_plot(engine, datasets, **kwargs):
    plt.figure(figsize=plt.figaspect(0.4))
    loss = xnn.TransformedNormalLoss(reduction='none')

    num_datasets = len(datasets)
    for (i, dataset) in enumerate(datasets):
        print(f"Dataset ({i+1}/{num_datasets})...")

        plt.subplot(2, num_datasets, i+1)
        plt.xlim((-2.5, 2.5))
        plt.ylim((-2.5, 2.5))
        plt.scatter(*dataset.T, s=1, color='orange')
        plt.xticks([])
        plt.yticks([])

        train(engine, dataset, **kwargs)
        plt.subplot(2, num_datasets, num_datasets+i+1)
        P.density_plot2d(lambda x: (-loss(*engine.model.eval()(x))).exp(), (-2.5, 2.5), (-2.5, 2.5))
        plt.xticks([])
        plt.yticks([])

        if i == num_datasets - 1:
            cbar = plt.colorbar(label='Density')
            cbar.set_ticks([])

    plt.tight_layout()
    plt.show()
