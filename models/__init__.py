# from .sashimi import Sashimi
from .wavenet import WaveNet

def construct_model(model_cfg):
    name = model_cfg.pop("_name_")
    model_cls = {
        "wavenet": WaveNet,
        # "sashimi": Sashimi,
    }[name]
    model = model_cls(**model_cfg)
    model_cfg["_name_"] = name # restore
    return model


def model_identifier(model_cfg):
    model_cls = {
        "melgen": WaveNet,
    }[model_cfg._name_]
    return model_cls.name(model_cfg)
