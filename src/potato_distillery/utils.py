from torch.nn import Module


def validate(model: Module, original_model: Module, verbose: bool, name: str):
    """
    Validates the parameters of the original and the distilled model.
    :param model: The distilled model.
    :param original_model: The original model.
    :param verbose: Whether to show additional info.
    :param name: The original model's name
    :return: None
    """
    for param_name in model.state_dict():
        sub_param, full_param = model.state_dict()[param_name], original_model.state_dict()[param_name]
        assert (sub_param.cpu().numpy() == full_param.cpu().numpy()).all(), param_name
        if verbose:
            print(f"[MODEL]{param_name} and [{name}]{param_name} are the same!\n")

    d_s = mem_calc(model)
    o_s = mem_calc(original_model)
    print(f"Model size reduced by: [Distilled]: {d_s} MB - [Original]: {o_s} MB [Saving]: {int(100 * d_s / o_s)}%\n")
    return


def mem_calc(model: Module):
    """
    Calculates the memory size of a model in MegaBytes(MiB)
    :param model: A torch Model.
    :return: The size in MiB
    """
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    return int(mem / (1024 * 1024))
