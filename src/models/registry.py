from importlib import import_module

_MODELS = {}


def register_model(name):
    def decorate(factory):
        if name in _MODELS:
            raise ValueError(f"Duplicate model {name}")
        _MODELS[name] = factory
        return factory
    return decorate


def list_models():
    for module in ("src.models.baselines", "src.models.scorecard", "src.models.block2"):
        import_module(module)
    return sorted(_MODELS)


def get_model(name):
    list_models()
    if name not in _MODELS:
        raise KeyError(f"Unknown model {name}. Registered: {sorted(_MODELS)}")
    return _MODELS[name]
