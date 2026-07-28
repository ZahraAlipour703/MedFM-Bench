DATASET_REGISTRY = {}


def register_dataset(name):
    """
    Decorator used to register datasets.
    """

    def decorator(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(
                f"Dataset '{name}' already registered."
            )

        DATASET_REGISTRY[name] = cls

        return cls

    return decorator


def build_dataset(name, **kwargs):

    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{name}' is not registered."
        )

    return DATASET_REGISTRY[name](**kwargs)