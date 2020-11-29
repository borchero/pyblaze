import torch
import sklearn.metrics as metrics

def accuracy(y_pred, y_true):
    """
    Computes the accuracy of the class predictions.

    Parameters
    ----------
    y_pred: torch.LongTensor [N] or torch.FloatTensor [N, C]
        The class predictions made by the model. Can be either specific classes or predictions for
        each class.
    y_true: torch.LongTensor [N] or torch.FloatTensor [N, C]
        The actual classes, either given as indices or one-hot vectors (more specifically, it may
        be any vector whose row-wise argmax values yield the class labels).

    Returns
    -------
    torch.FloatTensor
        The accuracy.
    """
    y_pred = _ensure_classes(y_pred)
    y_true = _ensure_classes(y_true)
    return (y_pred == y_true).float().mean()


def recall(y_pred, y_true, c=1):
    """
    Computes the recall score of the class predictions.

    Parameters
    ----------
    y_pred: torch.LongTensor [N] or torch.FloatTensor [N, C]
        The class predictions made by the model. Can be either specific classes or predictions for
        each class.
    y_true: torch.LongTensor [N]
        The actual classes.
    c: int, default: 1
        The class to calculate the recall score for. Default assumes a binary classification
        setting.

    Returns
    -------
    torch.FloatTensor
        The recall score.
    """
    y_pred = _ensure_classes(y_pred)

    y_pred = y_pred == c
    y_true = y_true == c

    correct = (y_pred[y_true == y_pred]).sum()
    true_correct = y_true.sum()

    return correct.float() / true_correct.float()


def precision(y_pred, y_true, c=1):
    """
    Computes the precision score of the class predictions.

    Parameters
    ----------
    y_pred: torch.LongTensor [N] or torch.FloatTensor [N, C]
        The class predictions made by the model. Can be either specific classes or predictions for
        each class.
    y_true: torch.LongTensor [N]
        The actual classes.
    c: int, default: 1
        The class to calculate the recall score for. Default assumes a binary classification
        setting.

    Returns
    -------
    torch.FloatTensor
        The precision score.
    """
    y_pred = _ensure_classes(y_pred)

    y_pred = y_pred == c
    y_true = y_true == c

    correct = (y_pred[y_true == y_pred]).sum()
    true_correct = y_pred.sum()

    return correct.float() / true_correct.float()


def f1_score(y_pred, y_true, c=1):
    """
    Computes the F1-score of the class predictions.

    Parameters
    ----------
    y_pred: torch.LongTensor [N] or torch.FloatTensor [N, C]
        The class predictions made by the model. Can be either specific classes or predictions for
        each class.
    y_true: torch.LongTensor [N]
        The actual classes.
    c: int, default: 1
        The class to calculate the recall score for. Default assumes a binary classification
        setting.

    Returns
    -------
    torch.FloatTensor
        The F1-score.
    """
    y_pred = _ensure_classes(y_pred)
    p = precision(y_pred, y_true, c=c)
    r = recall(y_pred, y_true, c=c)
    return (2 * p * r) / (p + r)


def roc_auc_score(y_pred, y_true):
    """
    Computes the area under the ROC curve.

    Parameters
    ----------
    y_pred: torch.FloatTensor [N]
        The (binary) predictions made by the model.
    y_true: torch.LongTensor [N]
        The actual classes.

    Returns
    -------
    torch.FloatTensor
        The ROC-AUC score.
    """
    assert y_pred.dim() == 1, \
        "ROC-AUC score only works in the binary case."

    return torch.as_tensor(
        metrics.roc_auc_score(y_true.numpy(), y_pred.numpy())
    )


def pr_auc_score(y_pred, y_true):
    """
    Computes the area under the precision-recall curve.

    Parameters
    ----------
    y_pred: torch.FloatTensor [N]
        The (binary) predictions made by the model.
    y_true: torch.LongTensor [N]
        The actual classes.

    Returns
    -------
    torch.FloatTensor
        The PR-AUC score.
    """
    prec, rec, _ = metrics.precision_recall_curve(y_true.numpy(), y_pred.numpy())
    auc = metrics.auc(rec, prec)
    return torch.as_tensor(auc)


def average_precision(y_pred, y_true):
    """
    Computes the average precision of the model predictions.

    Parameters
    ----------
    y_pred: torch.FloatTensor [N]
        The (binary) predictions made by the model.
    y_trye: torch.LongTensor [N]
        The actual classes.

    Returns
    -------
    torch.FloatTensor
        The average precision.
    """
    assert y_pred.dim() == 1, \
        "Average precision only work in the binary case."

    return torch.as_tensor(
        metrics.average_precision_score(y_true.numpy(), y_pred.numpy())
    )


def _ensure_classes(y):
    if y.dim() < 2:
        return y

    if y.dim() == 2:
        return torch.argmax(y, dim=-1)
    if y.dtype == torch.float:
        return torch.round(y)
    return y
