import copy
import functools
import inspect
import warnings

import torch
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin
import yaml

# Assumes the parent dir of pam is on PYTHONPATH
import pam

class CustomLinear(nn.Linear):
    # This class is for testing factory imports in submodules
    pass


class CustomConv2d(nn.Conv2d):
    # This class is for testing factory imports in submodules
    pass


class Bias(nn.Module):
    def __init__(self, num_features=None):
        super().__init__()
        self.num_features = num_features
        self.bias = Parameter(torch.empty(self.num_features))
        self.reset_parameters()

    def forward(self, x):
        shape = [dim if idx == 1 else 1 for idx, dim in enumerate(x.shape)]
        return x + self.bias.view(shape)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.bias)

    def extra_repr(self):
        return (
            "{num_features}".format(**self.__dict__)
        )


class LazyBias(LazyModuleMixin, Bias):
    cls_to_become = Bias  # type: ignore[assignment]

    def __init__(self):
        super().__init__(0)
        self.bias = UninitializedParameter()

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.num_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.num_features = input.shape[1]
                self.bias.materialize((self.num_features,))
                self.reset_parameters()


class Gain(nn.Module):
    def __init__(self, num_features=None):
        super().__init__()
        self.num_features = num_features
        self.gain = Parameter(torch.empty(self.num_features))
        self.reset_parameters()

    def forward(self, x):
        shape = [dim if idx == 1 else 1 for idx, dim in enumerate(x.shape)]
        return x * self.gain.view(shape)

    def reset_parameters(self):
        torch.nn.init.ones_(self.gain)

    def extra_repr(self):
        return (
            "{num_features}".format(**self.__dict__)
        )


class LazyGain(LazyModuleMixin, Gain):
    cls_to_become = Gain  # type: ignore[assignment]

    def __init__(self):
        super().__init__(0)
        self.gain = UninitializedParameter()

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.num_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.num_features = input.shape[1]
                self.gain.materialize((self.num_features,))
                self.reset_parameters()


def get_module_factory(desc, factories):
    if isinstance(desc, str):
        if desc in factories:
            desc = {'subtype': desc}
        else:
            try:
                desc = yaml.safe_load(desc)
            except Exception:
                raise ValueError(f'Unknown linear function {desc}')

    if isinstance(desc, dict):
        factory = factories[desc['subtype'].lower()]
        dka = desc.get('dkwargs', {})  # Default values, overwritten by arguments
        oka = desc.get('okwargs', {})  # Overwrite values, overwrite arguments
        return lambda *a, dka=dka, oka=oka, **ka: call_valid_kwargs(factory, {**dka, **ka, **oka}, a)
    elif isinstance(desc, list):
        return get_sequential_factory(desc)
    else:
        raise ValueError(f"Unknown configuration {desc=}")


def get_linear_factory(desc):
    desc = desc or 'standard'
    factories = {
        **dict.fromkeys(['standard', 'linear'], nn.Linear),
        'pam': pam.Linear,
        'custom': CustomLinear,
    }
    return get_module_factory(desc, factories)


def get_conv_factory(desc):
    desc = desc or 'standard'
    factories = {
        **dict.fromkeys(['standard', 'conv2d'], nn.Conv2d),
        'pam': pam.Conv2d,
        'custom': CustomConv2d,
    }
    return get_module_factory(desc, factories)


def get_bias_factory(desc):
    desc = desc or 'standard'
    factories = {
        **dict.fromkeys(['standard', 'bias'], Bias),
        'lazy_bias': LazyBias,
    }
    return get_module_factory(desc, factories)


def get_gain_factory(desc):
    desc = desc or 'standard'
    factories = {
        **dict.fromkeys(['standard', 'gain'], Gain),
        'lazy_bias': LazyGain,
    }
    return get_module_factory(desc, factories)


def get_activation_factory(desc):
    assert desc is not None, "Must specify activation type"
    factories = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
    }
    return get_module_factory(desc, factories)


def get_norm_factory(desc):
    assert desc is not None, "Must specify norm type"
    factories = {
        **dict.fromkeys(['batch_norm_2d', 'bn2d'], nn.BatchNorm2d),
        **dict.fromkeys(['layer_norm', 'ln'], nn.LayerNorm),
        **dict.fromkeys(['pam_layer_norm', 'pam_ln'], pam.LayerNorm),
    }
    return get_module_factory(desc, factories)


def get_sequential_factory(descriptions):
    factories = []
    for description in descriptions:
        if description['type'] == 'conv':
            factories.append(get_conv_factory(description))
        elif description['type'] == 'linear':
            factories.append(get_linear_factory(description))
        elif description['type'] == 'activation':
            factories.append(get_activation_factory(description))
        elif description['type'] == 'norm':
            factories.append(get_norm_factory(description))
        elif description['type'] == 'bias':
            factories.append(get_bias_factory(description))
        elif description['type'] == 'gain':
            factories.append(get_gain_factory(description))
        else:
            raise ValueError(f'Unknown factory type for {description}')

    return lambda factories=factories, **kwargs: torch.nn.Sequential(*[factory(**kwargs) for factory in factories])


def get_valid_kwargs(func, argdict):
    # https://stackoverflow.com/questions/196960/can-you-list-the-keyword-arguments-a-function-receives
    arguments = inspect.signature(func).parameters

    if any([p.kind == p.VAR_KEYWORD for p in arguments.values()]):
        # Accepts variable keyword arguments
        return argdict

    return {key: val for key, val in argdict.items() if key in set(arguments)}


def call_valid_kwargs(func, kwargs_dict, args_tuple=tuple()):
    kwargs = get_valid_kwargs(func, kwargs_dict)
    if extra_kwargs := sorted(set(kwargs_dict) - set(kwargs)):
        warnings.warn(f"Function {func} ignored extra arguments {extra_kwargs}")
    return func(*args_tuple, **kwargs)


def get_function_dict(desc):
    standard_dict = {
        'mean': torch.mean,
        'mul': torch.mul,
        'div': torch.div,
        'softmax': torch.nn.functional.softmax,
        'bmm': torch.bmm,
        'logsumexp': torch.logsumexp,
        'log_softmax': torch.nn.functional.log_softmax,
        'pow': torch.pow,
    }
    standard2_dict = {
        'mean': torch.mean,
        'mul': torch.mul,
        'div': torch.div,
        'softmax': pam.pam_experimental.torch_softmax2,
        'bmm': torch.bmm,
        'logsumexp': pam.pam_experimental.torch_logsumexp2,
        'log_softmax': pam.pam_experimental.torch_log_softmax2,
        'pow': torch.pow,
    }
    pam_dict = {
        'mean': pam.mean,
        'mul': pam.mul,
        'div': pam.div,
        'softmax': pam.softmax,
        'bmm': pam.bmm,
        'logsumexp': pam.logsumexp,
        'log_softmax': pam.log_softmax,
        'pow': pam.pow,
    }
    pam_dict_bmm_only = {
        'mean': torch.mean,
        'mul': torch.mul,
        'div': torch.div,
        'softmax': torch.nn.functional.softmax,
        'bmm': pam.bmm,
        'logsumexp': torch.logsumexp,
        'log_softmax': torch.nn.functional.log_softmax,
        'pow': torch.pow,
    }
    pam_dict_softmax_only = {
        'mean': torch.mean,
        'mul': torch.mul,
        'div': torch.div,
        'softmax': pam.softmax,
        'bmm': torch.bmm,
        'logsumexp': torch.logsumexp,
        'log_softmax': torch.nn.functional.log_softmax,
        'pow': torch.pow,
    }

    if isinstance(desc, str):
        if desc in ['standard', 'torch']:
            return copy.copy(standard_dict)
        elif desc == 'standard2':
            return copy.copy(standard2_dict)
        elif desc == 'pam':
            return copy.copy(pam_dict)
        elif desc == 'pam_bmm_only':
            return copy.copy(pam_dict_bmm_only)
        elif desc == 'pam_softmax_only':
            return copy.copy(pam_dict_softmax_only)
        else:
            try:
                decoded = yaml.safe_load(desc)
                desc = decoded
            except Exception:
                raise ValueError(f'Unknown function config {desc}')

    assert isinstance(desc, dict), f"Unknown function cfg {desc=}"
    assert desc['type'] in ['pam','pam_bmm_only','pam_softmax_only'], "Only support custom cfg for pam"
    dkwargs = desc.get('dkwargs', {})
    okwargs = desc.get('okwargs', {})
    return {
        name: lambda *a, func=func, dkwargs=dkwargs, okwargs=okwargs, **ka: 
        call_valid_kwargs(func, {**dkwargs, **ka, **okwargs}, a)
        for name, func in pam_dict.items()
    }
