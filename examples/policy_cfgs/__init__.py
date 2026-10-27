__all__ = ["Pi05Cfg", "Pi05LerobotCfg", "Pi05LerobotJointCfg", "SmolVLACfg"]


def __getattr__(name):
    if name == "Pi05Cfg":
        from .pi05 import Pi05Cfg

        return Pi05Cfg
    if name == "Pi05LerobotCfg":
        from .pi05_lerobot import Pi05LerobotCfg

        return Pi05LerobotCfg
    if name == "Pi05LerobotJointCfg":
        from .pi05_lerobot_joint import Pi05LerobotJointCfg

        return Pi05LerobotJointCfg
    if name == "SmolVLACfg":
        from .smolvla import SmolVLACfg

        return SmolVLACfg
    raise AttributeError(f"module 'examples.policy_cfgs' has no attribute {name!r}")
