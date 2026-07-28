class Registry:
    """
    Generic registry for datasets, models, metrics,
    transforms, losses, etc.
    """

    def __init__(self, name):
        self.name = name
        self._registry = {}

    def register(self, name=None):

        def decorator(obj):

            key = name or obj.__name__

            if key in self._registry:
                raise KeyError(
                    f"{key} already registered in {self.name}"
                )

            self._registry[key] = obj

            return obj

        return decorator

    def build(self, name, **kwargs):

        if name not in self._registry:

            raise KeyError(
                f"{name} is not registered in {self.name}"
            )

        return self._registry[name](**kwargs)

    def get(self, name):

        return self._registry[name]

    def keys(self):

        return list(self._registry.keys())