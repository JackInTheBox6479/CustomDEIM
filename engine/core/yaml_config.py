import copy

from yaml_utils import *

from engine.core.config import BaseConfig


class YAMLConfig(BaseConfig):
    def __init__(self, config_path, **kwargs):
        super().__init__()

        cfg = load_config(config_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_config = copy.deepcopy(cfg)

        for k in super().__dict__:
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]
