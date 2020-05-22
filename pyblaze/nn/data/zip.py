class ZipDataLoader:
    """
    A data loader that zips together two underlying data loaders. The data loaders must be
    sampling the same batch size and :code:`drop_last` must be set to `True` on data loaders that
    sample from a fixed-size dataset. Whenever one of the data loaders has a fixed size, this data
    loader defines a length. This length is given as the minimum of the both lengths divided by
    their respective counts.

    A common use case for this class are Wasserstein GANs where the critic is trained for multiple
    iterations for each data batch.
    """

    def __init__(self, lhs_loader, rhs_loader, lhs_count=1, rhs_count=1):
        """
        Initializes a new data loader.

        Parameters
        ----------
        lhs_dataset: torch.utils.data.DataLoader
            The dataset to sample from for the first item of the data tuple.
        rhs_dataset: torch.utils.data.DataLoader
            The dataset to sample from for the second item of the data tuple.
        lhs_count: int
            The number of items to sample for the first item of the data tuple.
        rhs_count: int
            The number of items to sample for the second item of the data tuple.
        """
        if lhs_loader.batch_size != rhs_loader.batch_size:
            raise ValueError("Both given data loaders must have the same batch size.")

        self.lhs_loader = lhs_loader
        self.rhs_loader = rhs_loader
        self.lhs_count = lhs_count
        self.rhs_count = rhs_count

    def __len__(self):
        result = None

        try:
            result = len(self.lhs_loader) // self.lhs_count
        except:  # pylint: disable=bare-except
            pass

        try:
            rhs_len = len(self.rhs_loader) // self.rhs_count
            if result is None:
                result = rhs_len
            else:
                result = min(result, rhs_len)
        except:  # pylint: disable=bare-except
            pass

        if result is None:
            raise TypeError("__len__ not implemented for instance of ZipDataLoader")

        return result

    def __iter__(self):
        lhs_it = iter(self.lhs_loader)
        rhs_it = iter(self.rhs_loader)
        while True:
            try:
                lhs_items = [next(lhs_it) for _ in range(self.lhs_count)]
                rhs_items = [next(rhs_it) for _ in range(self.rhs_count)]
            except StopIteration:
                return

            yield lhs_items, rhs_items
