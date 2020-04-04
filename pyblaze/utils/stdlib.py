class cached_property:
    """
    Python property which is computed exactly once and a cached value is returned upon every
    subsequent call.
    """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def __get__(self, obj, objtype):
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


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
