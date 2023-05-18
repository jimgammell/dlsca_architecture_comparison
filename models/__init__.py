from collections import OrderedDict

from models.cnn_baselines import ProuffCnn, WoutersNet

AVAILABLE_MODELS = OrderedDict([
    (model_class.model_name, model_class)
    for model_class in [ProuffCnn, WoutersNet]
])

def get_available_models():
    return list(AVAILABLE_MODELS.keys())

def check_name(model_name):
    if not model_name in get_available_models():
        raise ValueError('Unrecognized model name: \'{}\''.format(model_name))

def get_class(model_name):
    check_name(model_name)
    return AVAILABLE_MODELS[model_name]