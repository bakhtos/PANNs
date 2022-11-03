import torch

__all__ = ['forward']


@torch.no_grad()
def forward(model, data_loader, return_data=False,
            return_target=False):
    """Forward data to a model.

    Parameters
    __________

    model : torch.nn.Module subclass,
        Model to forward the data to.
    data_loader: torch.utils.data.Dataloader,
        Data loader that provides data and target to the model
    return_data: bool, optional (default False)
        If True, third returned argument will be collated input data
    return_target: bool, optional (default False)
        If, True, fourth returned argument will be collated target

    Returns
    _______
    clipwise_output : torch.Tensor,
        First output of the model, tensor of shape (audios_num, classes_num)
    second_output : torch.Tensor,
        Either segmentwise or framewise output,
        tensor of shape(audios_num, segments_num or frames_num, classes_num)
    data : torch.Tensor,
        If return_data is True, tensor of shape (audios_num, clip_samples),
        otherwise None
    target: torch.Tensor,
        If return_target is True, tensor of shape (audios_num, classes_num),
        otherwise None
    """

    # Forward data to a model in mini-batches

    all_clipwise_output = []
    all_second_output = []
    all_data = []
    all_target = []
    for data, target in data_loader:
        clipwise_output, second_output = model(data)

        all_clipwise_output.append(clipwise_output)
        all_second_output.append(second_output)

        if return_data:
            all_data.append(data)
        if return_target:
            all_target.append(target)

    all_clipwise_output = torch.cat(all_clipwise_output, dim=0)
    all_second_output = torch.cat(all_second_output, dim=0)
    all_data = torch.cat(all_data) if return_data else None
    all_target = torch.cat(all_target) if return_target else None

    return all_clipwise_output, all_second_output, all_data, all_target
