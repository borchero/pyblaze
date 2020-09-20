from torch.utils.data import Dataset

class TransformDataset(Dataset):
    """
    A dataset which allows for mapping a dataset's items using a series of transformations.
    """

    def __init__(self, source, *transforms):
        """
        Initializes a new dataset using the given transformations.

        Parameters
        ----------
        source: torch.utils.data.Dataset
            The base dataset.
        transforms: varargs of callable (object) -> object
            The transforms to apply. The first transform receives as input a source dataset's item.
            Subsequent transforms receive as input the output of the previous transform. This
            dataset hence outputs the output of the last transform.
        """
        super().__init__()

        self.source = source
        self.transforms = transforms

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        item = self.source[index]
        for transform in self.transforms:
            item = transform(item)
        return item
