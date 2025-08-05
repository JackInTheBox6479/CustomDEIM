import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T

from typing import Any, Dict, List, Optional

from ._transforms import EmptyTransform
from ...core import register, GLOBAL_CONFIG
torchvision.disable_beta_transforms_warning()
import random


@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None, mosaic_prob=-0.1) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transform = getattr(GLOBAL_CONFIG[name]['_pymodule'], GLOBAL_CONFIG[name]['_name'])(**op)
                    transforms.append(transform)
                    op['type'] = name
                    print("     ### Transform @{} ###    ".format(type(transform).__name__))

                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]

        super().__init__(transforms=transforms)

        self.mosaic_prob = mosaic_prob
        if policy is None:
            policy = {'name': 'default'}
        else:
            if self.mosaic_prob > 0:
                print("     ### Mosaic with Prob.@{} and ZoomOut/IoUCrop existed ### ".format(self.mosaic_prob))
            print("     ### ImgTransforms Epochs: {} ### ".format(policy['epoch']))
            print('     ### Policy_ops@{} ###'.format(policy['ops']))
        self.global_samples = 0
        self.policy = policy

    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy['name'])(*inputs)

    def get_forward(self, name):
        forwards = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
            'stop_sample': self.stop_sample_forward,
        }
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_epoch = self.policy['epoch']

        if isinstance(policy_epoch, list) and len(policy_epoch) == 3:
            if policy_epoch[0] <= cur_epoch < policy_epoch[1]:
                with_mosaic = random.random() <= self.mosaic_prob
                with_mosaic = False
            for transform in self.transforms:
                if (type(transform).__name__ in policy_ops and cur_epoch < policy_epoch[0]):
                    pass
                elif (type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch[-1]):
                    pass
                else:
                    if (type(transform).__name__ == 'Mosaic' and not with_mosaic):
                        pass
                    elif (type(transform).__name__ == 'RandomZoomOut' or type(transform).__name__ == 'RandomIoUCrop') and with_mosaic:
                        pass
                    else:
                        sample = transform(sample)
        else:
            for transform in self.transforms:
                if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                    pass
                else:
                    sample = transform(sample)

        return sample


    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_sample = self.policy['sample']

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample
