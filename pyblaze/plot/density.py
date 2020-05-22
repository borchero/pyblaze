import torch
import matplotlib.pyplot as plt

def density_plot2d(distribution, x_range=(-1, 1), y_range=(-1, 1), resolution=500,
                   cmap='Blues', **kwargs):
    """
    Generates a scatter plot visualizing the distribution's density in the given 2D region.

    Parameters
    ----------
    distribution: torch.nn.Module
        The distribution modeled as PyTorch module. Must take a `torch.Tensor [N, 2]` (number of
        evaluation points N) as input and return the log-probabilities as `torch.Tensor [N]`.
    x_range: tuple of (float, float), default: (-1, 1)
        The range to visualize in x-dimension.
    y_range: tuple of (float, float), default: (-1, 1)
        The range to visualize in y-dimension.
    resolution: int, default: 500
        The number of evaluation points for each dimension. The total number of evaluation points
        is therefore given by the squared resolution.
    cmap: str, default: 'gist_heat'
        The matplotlib colorbar to use for visualization.
    kwargs: keyword arguments
        Additional arguments passed to the :code:`scatter` method.
    """
    x, y = torch.meshgrid(
        torch.linspace(*x_range, steps=resolution),
        torch.linspace(*y_range, steps=resolution)
    )

    measure_points = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    with torch.no_grad():
        log_probs = distribution(measure_points)

    vis_points = measure_points.numpy().T
    plt.scatter(*vis_points, c=log_probs.exp().numpy(), cmap=cmap, **kwargs)
