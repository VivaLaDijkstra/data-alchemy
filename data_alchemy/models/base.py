from data_alchemy.utils.auto_resume import Cacheable


class BaseModel(Cacheable):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the __init__ method.")
