from collections import OrderedDict

from models.resnet import ResNet
from models.cnn_baselines import ProuffCnn, WoutersNet
from models.convmixer import ConvMixer
from models.vgg import VGG
from models.wide_resnet import WideResNet

AVAILABLE_MODELS = OrderedDict([
    (model_class.model_name, model_class)
    for model_class in [ProuffCnn, WoutersNet, ResNet, ConvMixer, VGG, WideResNet]
])

def get_available_models():
    return list(AVAILABLE_MODELS.keys())

def check_name(model_name):
    if not model_name in get_available_models():
        raise ValueError('Unrecognized model name: \'{}\''.format(model_name))

def get_class(model_name):
    check_name(model_name)
    return AVAILABLE_MODELS[model_name]