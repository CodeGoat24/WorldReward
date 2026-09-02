from importlib import import_module

__all__ = ["RewardDispatcher", "RewardMetrics", "RewardRequest"]

_EXPORTS = {
    "RewardDispatcher": (".reward_dispatcher", "RewardDispatcher"),
    "RewardMetrics": (".utils.types", "RewardMetrics"),
    "RewardRequest": (".utils.types", "RewardRequest"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
