"""
Code from https://github.com/gmum/proto-segmentation.

NO Modifications.
"""
from typing import Any, Dict

import gin


def sanitize_config_val_for_json(val: Any) -> Any:
    if isinstance(val, list):
        return [sanitize_config_val_for_json(v) for v in val]
    if isinstance(val, dict):
        return {k: sanitize_config_val_for_json(v) for k, v in val.items()}
    return val


def get_operative_config_json() -> Dict:
    json_gin_config = {}
    # noinspection PyProtectedMember
    for configurable_key, params in gin.config._OPERATIVE_CONFIG.items():
        configurable_key = ''.join(configurable_key)
        for param_key, param_val in params.items():
            full_key = '.'.join((configurable_key, param_key))
            param_val = sanitize_config_val_for_json(param_val)
            json_gin_config[full_key] = param_val
    return json_gin_config
