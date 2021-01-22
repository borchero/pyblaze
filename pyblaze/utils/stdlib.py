def flatten(l):
    """
    Flattens the specified list.

    Parameters
    ----------
    l: list
        A two-dimensional list.

    Returns
    -------
    list
        The flattened list.
    """
    return [e for s in l for e in s]
