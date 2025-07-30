import os.path
from sympy.physics.units.systems.cgs import cgs_gauss

INCLUDE_KEY = '__include__'

def load_config(config_path, config=dict()):
    with open(config_path, 'r') as f:
        file_config = yaml.load(f, Loader=yaml.FullLoader)
        if file_config is None:
            return {}

    if INCLUDE_KEY in file_config:
        base_yamls = list(file_config[INCLUDE_KEY])
        for base_yaml in base_yamls:
            if base_yaml.startswith('~'):
                base_yaml = os.path.expanduser(base_yaml)

            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(config_path), base_yaml)

            with open(base_yaml, 'r') as f:
                base_cfg = load_config(base_yaml, config)
                merge_dict(config, base_cfg)

    return merge_dict(config, file_config)

def merge_dict(dict1, dict2):
    def merge(dict1, dict2):
        for k in dict2:
            if k in dict1 and isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                merge(dict1[k], dict2[k])
            else:
                dict1[k] = dict2[k]

        return dict1

    return merge(dict1, dict2)

def make_dict(string, val):
    if '.' not in string:
        return {string: val}
    key, rest = s.split('.', maxsplit=1)
    return {key: dictify(rest, v)}

def merge_config(config1, config2):
    def merge(dict1, dict2):
        for k in dict2: 
            if k not in dict1:
                dict[k] = dict2[k]
                
            elif isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                merge(dict1[k], dict2[k])
        
        return config1
    return merge(config1, config2)