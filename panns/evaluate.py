from sklearn.metrics import average_precision_score, roc_auc_score

from .forward import forward


def evaluate(model, data_loader):
    """Forward evaluation data to the model and calculate AP and AUC metrics.

    Args:
        model: torch.nn.Module subclass, Torch model to be evaluated
        data_loader: torch.utils.data.Dataloader, Data loader for the evaluation data

    Returns:
    average_precision, auc
        -Average precision score metric result from sklearn
        -Area under curve metric result from sklearn
    """

    # Forward
    forward_output = forward(model=model, data_loader=data_loader,
                             return_target=True)
    # TODO Change to .numpy(force=True) when torch 1.11 is supported
    clipwise_output = forward_output.clipwise_output.cpu().numpy()
    target = forward_output.target.cpu().numpy()

    average_precision = average_precision_score(target, clipwise_output,
                                                average=None)

    auc = roc_auc_score(target, clipwise_output, average=None)
    
    return average_precision, auc
