def forward(model, x, **kwargs):
    """
    Computes the model output for common types of inputs.

    Parameters
    ----------
    model: torch.nn.Module
        The model to compute the output for.
    x: list or tuple or dict or object
        The input the model.
    kwargs: keyword arguments
        Additional model inputs passed by keyword.

    Returns
    -------
    object
        The output of the model (although arbitrary, it is usually a torch.Tensor).
    """
    if isinstance(x, (list, tuple)):
        return model(*x, **kwargs)
    if isinstance(x, dict):
        return model(**x, **kwargs)
    return model(x, **kwargs)
