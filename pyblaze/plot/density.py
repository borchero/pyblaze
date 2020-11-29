import torch
import matplotlib.pyplot as plt

def density_plot2d(callback, x_range=(-1, 1), y_range=(-1, 1), resolution=500,
                   cmap='Blues', threshold=None, ax=None, **kwargs):
    """
    Generates a scatter plot visualizing a distribution's density in the given 2D region.

    Parameters
    ----------
    callback: callable
        The distribution for which the probability is evaluated. Potentially given as PyTorch
        module. Must take a `torch.Tensor [N, 2]` (number of evaluation points N) as input and
        return the output as `torch.Tensor [N]`.
    x_range: tuple of (float, float), default: (-1, 1)
        The range to visualize in x-dimension.
    y_range: tuple of (float, float), default: (-1, 1)
        The range to visualize in y-dimension.
    resolution: int, default: 500
        The number of evaluation points for each dimension. The total number of evaluation points
        is therefore given by the squared resolution.
    cmap: str, default: 'gist_heat'
        The matplotlib colorbar to use for visualization.
    threshold: float, default: None
        A minimum value that needs to be surpassed in order for a scatter point to be plotted.
    ax: matplotlib.axes, default: None
        The axis to use for plotting or None if the global imperative API of matplotlib should be
        used.
    kwargs: keyword arguments
        Additional arguments passed to the :code:`scatter` method.
    """
    x, y = torch.meshgrid(
        torch.linspace(*x_range, steps=resolution),
        torch.linspace(*y_range, steps=resolution)
    )

    measure_points = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    with torch.no_grad():
        values = callback(measure_points).numpy()

    if threshold is not None:
        mask = values > threshold
    else:
        mask = torch.ones(values.shape[0], dtype=torch.bool)

    vis_points = measure_points[mask].numpy().T
    if ax is not None:
        return ax.scatter(*vis_points, c=values[mask], cmap=cmap, **kwargs)
    return plt.scatter(*vis_points, c=values[mask], cmap=cmap, **kwargs)
