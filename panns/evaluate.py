from sklearn import metrics

from .forward import forward

def evaluate(model, data_loader):
    """Forward evaluation data and calculate statistics.

    Args:
      data_loader: object

    Returns:
      statistics: dict, 
          {'average_precision': (classes_num,), 'auc': (classes_num,)}
    """

    # Forward
    output_dict = forward(
        model=model, 
        generator=data_loader, 
        return_target=True)

    clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
    target = output_dict['target']    # (audios_num, classes_num)

    average_precision = metrics.average_precision_score(
        target, clipwise_output, average=None)

    auc = metrics.roc_auc_score(target, clipwise_output, average=None)
    
    return average_precision, auc
