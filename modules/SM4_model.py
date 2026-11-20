import torch

from modules.SM3_model import HDCSModel
from modules.HDCS_model import ConfigurableHDCS


def _four_scenario_options(opt):
    """Return a copy so callers cannot accidentally configure SM4 with K != 4."""
    options = dict(opt)
    options["domain_num"] = 4
    return options


class MaskDNN(HDCSModel):
    """Four-scenario HDCS with a DNN backbone."""
    def __init__(self, opt):
        super().__init__(_four_scenario_options(opt), backbone="dnn")


class ConfiguredMaskDNN(ConfigurableHDCS):
    """Opt-in AliMama adaptation with the same HDCS selection/allocation paths."""
    def __init__(self, opt):
        super().__init__(dict(_four_scenario_options(opt), model='dnn'))


class MaskDeepFM(HDCSModel):
    """Four-scenario HDCS with a DeepFM backbone."""
    def __init__(self, opt):
        super().__init__(_four_scenario_options(opt), backbone="deepfm")


class MaskDeepCross(HDCSModel):
    """Four-scenario HDCS with a DCN backbone."""
    def __init__(self, opt):
        super().__init__(_four_scenario_options(opt), backbone="dcn")


def getOptim(network, optim, lr, l2):
    optimizer_name = optim.lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(network.parameters(), lr=lr, weight_decay=l2)
    if optimizer_name == "adam":
        return torch.optim.Adam(network.parameters(), lr=lr, weight_decay=l2)
    raise ValueError(f"Invalid optimizer type: {optim}")


def getModel(model: str, opt):
    model_name = model.lower()
    if model_name == "deepfm":
        return MaskDeepFM(opt)
    if model_name == "dcn":
        return MaskDeepCross(opt)
    if model_name == "dnn":
        if opt.get('configured', False):
            return ConfiguredMaskDNN(opt)
        return MaskDNN(opt)
    raise ValueError(f"Invalid model type: {model}")
