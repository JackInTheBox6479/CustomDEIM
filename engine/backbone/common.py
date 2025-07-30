import torch
import torch.nn as nn

class ConvNormLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size-1)//2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.Identity() if act is None else get_activation(act)

        def forward(self, x):
            return self.act(self.norm(self.conv(x)))

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, epsilon=1e-6):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.epsilon = epsilon
        self.num_features = num_features

    def load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self).load_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.epsilon).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    #TODO: Figure out what this means
    def extra_repr(self):
        return('num_features={}, bias={}, running_mean={}, running_var={}').format(**self.__dict__)

def freeze_bn(module):
    if isinstance(module, nn.BatchNorm2d):
        module = FrozenBatchNorm2d(module.num_features)
    else:
        for name, child in module.named_children():
            _child = freeze_bn(child)
            if _child is not child:
                setattr(module, name, _child)
    return module


def get_activation(act, inplace=True):
    if act is None:
        return nn.Identity()
    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == 'silu' or act == 'swish':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'silu':
        m = nn.SiLU()
    elif act == 'gelu':
        m = nn.GELU()
    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError('')
    if hasattr(m, 'inplace'):
        m.inplace = inplace

    return m
