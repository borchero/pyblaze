import torch.distributions as dist

def generate_noise(shape, noise_type='normal'):
    """
    Generate noise which is usually fed to the generator of a GAN.

    Parameters
    ----------
    shape: torch.Size
        The shape of the noise to generate.
    noise_type: str, default: 'normal'
        The type of the noise to generate. Must be one of the following:

        * normal: Draws from a standard normal distribution.
        * uniform: Draws uniformly from the range [-1, 1].

    Returns
    -------
    torch.Tensor
        The generated noise with the specified size.
    """
    if noise_type == 'normal':
        return dist.Normal(0, 1).sample(shape)
    if noise_type == 'uniform':
        return dist.Uniform(-1, 1).sample(shape)
    raise ValueError(f'Invalid noise type {noise_type}.')
    