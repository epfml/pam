import torch

def to_tensor(*values, device=None):
    for element in values:
        if isinstance(element, torch.Tensor):
            assert device is None or device == element.device, \
                f"Mismatching devices {device=}, {element.device=}"
            device = element.device

    out_tensors = []
    for element in values:
        if isinstance(element, (int, float)):
            element = torch.tensor(element, device=device, dtype=torch.float32)
        out_tensors.append(element)

    return out_tensors


def broadcast_tensors(*tensors):
    return torch.broadcast_tensors(*to_tensor(*tensors))


def demote_tensor(out_value, in_values):
    if all(isinstance(v, (float, int)) for v in in_values):
        return out_value.item()
    else:
        return out_value
