import importlib


def make_model(name: str):
    base = importlib.import_module('models.' + name)
    if name.startswith('twostage'):
        model = getattr(base, 'TwoStageERRNetModel')
    else:
        model = getattr(base, 'YTMTNetModel')
    return model
