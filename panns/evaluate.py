from sklearn.metrics import average_precision_score, roc_auc_score

from .forward import forward


def evaluate(model, data_loader):
    """Forward evaluation data to the model and calculate AP and AUC metrics.

    Parameters
    __________

    model : torch.nn.Module subclass,
        Torch model to be evaluated
    data_loader: torch.utils.data.Dataloader,
        Data loader for the evaluation data

    Returns
    _______

    average_precision : float,
        Average precision score metric result from sklearn
    auc: float,
        Area under curve metric result from sklearn
    """

    # Forward
    clipwise_output, _, _, target = forward(model=model, data_loader=data_loader,
                                            return_target=True)
    clipwise_output = clipwise_output.numpy(force=True)
    target = target.numpy(force=True)

    average_precision = average_precision_score(target, clipwise_output,
                                                average=None)

    auc = roc_auc_score(target, clipwise_output, average=None)
    
    return average_precision, auc
