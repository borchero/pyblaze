from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate

# pylint: disable=abstract-method
class ZipDataset(IterableDataset):
    """
    An infinite dataset that samples a list of items from two infinite datasets. Usually used with
    Wasserstein GANs where the critic is trained for multiple iterations.
    """

    def __init__(self, lhs_dataset, rhs_dataset, lhs_count=1, rhs_count=1):
        """
        Initializes a new dataset.

        Parameters
        ----------
        lhs_dataset: torch.utils.data.IterableDataset
            The dataset to sample from for the first item of the data tuple.
        rhs_dataset: torch.utils.data.IterableDataset
            The dataset to sample from for the second item of the data tuple.
        lhs_count: int
            The number of items to sample for the first item of the data tuple.
        rhs_count: int
            The number of items to sample for the second item of the data tuple.
        """
        super().__init__()

        self.lhs = lhs_dataset
        self.rhs = rhs_dataset
        self.lhs_count = lhs_count
        self.rhs_count = rhs_count

    def __iter__(self):
        lhs_iter = iter(self.lhs)
        rhs_iter = iter(self.rhs)
        while True:
            # pylint: disable=stop-iteration-return
            lhs = [next(lhs_iter) for _ in range(self.lhs_count)]
            rhs = [next(rhs_iter) for _ in range(self.rhs_count)]
            yield lhs, rhs


class ZipDatasetCollater:
    """
    An object that knows how to collate a zip dataset by knowing how to collate the individual
    items.
    """

    def __init__(self, collate_lhs=default_collate, collate_rhs=default_collate):
        """
        Initializes a new collater.

        Parameters
        ----------
        collate_lhs: callable, default: PyTorch default collation
            The collation function for the first item of the data tuple.
        collate_rhs: callable, default: PyTorch default collation
            The collation function for the second item of the data tuple.
        """
        self.collate_lhs = collate_lhs
        self.collate_rhs = collate_rhs

    def collate(self, items):
        """
        Collates the items obtained from some zip dataset.
        """
        lhs_items = [item[0] for item in items]
        rhs_items = [item[1] for item in items]

        lhs_count = len(lhs_items[0])
        lhs_out = [self.collate_lhs([item[i] for item in lhs_items]) for i in range(lhs_count)]

        rhs_count = len(rhs_items[0])
        rhs_out = [self.collate_lhs([item[i] for item in rhs_items]) for i in range(rhs_count)]

        return lhs_out, rhs_out
