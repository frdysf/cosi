from typing import Callable
from torch.nn import Module
from torch.nn.modules import activation

Activation = Callable[..., Module]

def get_activation_fn(act: str) -> Activation:
    '''
    Helper function to get activation function from string.
        https://discuss.pytorch.org/t/call-activation-function-from-string/30857/5
    '''
    # get list from activation submodule as lower-case
    activations_lc = [str(a).lower() for a in activation.__all__]
    if (act := str(act).lower()) in activations_lc:
        # match actual name from lower-case list, return function/factory
        idx = activations_lc.index(act)
        act_name = activation.__all__[idx]
        act_func = getattr(activation, act_name)
        return act_func
    else:
        raise ValueError(f'Cannot find activation function for string <{act}>')
    

from torch.nn import init

def get_initialization_fn(init_: str) -> callable:
    '''
    Helper function to get initialization function from string.
    '''
    # get list from init submodule as lower-case
    inits_lc = [str(i).lower() for i in init.__dict__]
    if (init_ := str(init_).lower()) in inits_lc:
        # match actual name from lower-case list, return function/factory
        init_func = getattr(init, init_)
        return init_func
    else:
        raise ValueError(f'Cannot find initialization function for string <{init_}>')


from torch.nn.modules import loss

Loss = Callable[..., Module]

def get_loss_fn(loss_: str) -> Loss:
    '''
    Helper function to get loss function from string.
    '''
    # get list from activation submodule as lower-case
    losses_lc = [str(a).lower() for a in loss.__all__]
    if (loss_ := str(loss_).lower()) in losses_lc:
        # match actual name from lower-case list, return function/factory
        idx = losses_lc.index(loss_)
        loss_name = loss.__all__[idx]
        loss_func = getattr(loss, loss_name)
        return loss_func
    else:
        raise ValueError(f'Cannot find loss function for string <{loss_}>')