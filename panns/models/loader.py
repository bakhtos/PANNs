import torch

import panns.models.models

__all__ = ['load_model']


def load_model(model, checkpoint=None, **kwargs):
    """
    
    Args:
        model: Name of model class from panns.models.models.
        checkpoint: Path to saved checkpoint (state_dict) of the model
                    to be loaded (default None).
        **kwargs: Other keyword arguments for the model.

    Returns: model
    """

    if model in panns.models.models.__all__:
        model = eval("panns.models.models."+model)
    else:
        raise ValueError(f"'{model}' is not among the defined models.")

    model = model(**kwargs)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))

    return model
