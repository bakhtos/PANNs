from collections import namedtuple

import torch

__all__ = ['ForwardOutput', 'forward']

ForwardOutput = namedtuple('ForwardOutput', ('clipwise_output',
                                             'segmentwise_output',
                                             'framewise_output', 'embedding',
                                             'data', 'target'))


@torch.no_grad()
def forward(model, data_loader, return_data=False,
            return_target=False):
    """Forward data to a model.

    Output is return as namedtuple.

    Args:
        model : torch.nn.Module subclass, Model to forward the data to.
        data_loader: torch.utils.data.Dataloader,
            Data loader that provides data and target to the model
        return_data: bool, If True, fifth returned argument will be
            collated input data (default False)
        return_target: bool, If True, sixth returned argument will be
            collated target (default False)

    Returns:
        clipwise_output, segmentwise_output, framewise_output, embedding, data,
        target
    """

    # Forward data to a model in mini-batches

    all_clipwise_output = []
    all_segmentwise_output = []
    all_framewise_output = []
    all_embedding = []
    all_data = []
    all_target = []
    for data, target in data_loader:
        clipwise_output, segmentwise_output, framewise_output, embedding = \
            model(data)

        all_clipwise_output.append(clipwise_output)
        all_segmentwise_output.append(segmentwise_output)
        all_framewise_output.append(framewise_output)
        all_embedding.append(embedding)

        if return_data:
            all_data.append(data)
        if return_target:
            all_target.append(target)

    all_clipwise_output = torch.cat(all_clipwise_output, dim=0)
    all_segmentwise_output = torch.cat(all_segmentwise_output, dim=0)
    all_framewise_output = torch.cat(all_framewise_output, dim=0)
    all_embedding = torch.cat(all_embedding, dim=0)
    all_data = torch.cat(all_data) if return_data else None
    all_target = torch.cat(all_target) if return_target else None

    return ForwardOutput(all_clipwise_output, all_segmentwise_output,
                         all_framewise_output, all_embedding, all_data,
                         all_target)
