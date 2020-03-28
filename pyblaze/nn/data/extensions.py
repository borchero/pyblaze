import math
import numpy as np
import torch.utils.data as data

###############
### METHODS ###
###############
def loader(self, **kwargs):
    """
    Returns a data loader for this dataset. If the dataset defines a `collate_fn` function, this is
    automatically set. When `pyblaze.nn` is imported, this method is available on all
    `torch.utils.data.Dataset` objects.

    Parameters
    ----------
    kwargs: keyword arguments
        Paramaters passed directly to the DataLoader.

    Returns
    -------
    torch.utils.data.DataLoader
        The data loader with the specified attributes.
    """
    if hasattr(self, 'collate_fn'):
        kwargs['collate_fn'] = self.collate_fn
    return data.DataLoader(self, **kwargs)


def random_split(self, *sizes, seed=None):
    """
    Splits the dataset randomly into multiple subsets. When `pyblaze.nn` is imported, this method is
    available on all `torch.utils.data.Dataset` objects.

    Parameters
    ----------
    sizes: variadic argument of float
        The sizes of the splits, given as fraction of the size of the dataset. Hence, the sizes
        must sum to 1.
    seed: int, default: None
        If given, uses the specified seed to sample the indices for each subset.

    Returns
    -------
    list of torch.utils.data.Subset
        The random splits of this dataset.
    """
    assert math.isclose(sum(sizes), 1), \
        "Sizes do not sum to 1."

    # pylint: disable=no-member
    randomizer = np.random.RandomState(seed)

    # Get subset sizes
    nums = []
    for i, size in enumerate(sizes):
        if i == len(sizes) - 1:
            nums.append(len(self) - sum(nums))
        else:
            nums.append(int(np.round(size * len(self))))

    # Get subset indices
    indices = randomizer.permutation(len(self))
    index_choices = []
    c = 0
    for num in nums:
        index_choices.append(indices[c:c+num])
        c += num

    return [
        data.Subset(self, indices) for indices in index_choices
    ]

##################
### EXTENSIONS ###
##################
data.Dataset.loader = loader
data.Dataset.random_split = random_split
