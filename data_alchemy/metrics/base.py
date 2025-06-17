class BaseMetric:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
