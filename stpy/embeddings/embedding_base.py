from abc import ABC, abstractmethod
from typing import  Union
from stpy.borel_set import BorelSet

class EmbeddingBase(ABC):
    """
    Base class implementing finite dimensional embedding
    """
    def __init__(self):
        pass

    @abstractmethod
    def integral(self, S: BorelSet):
        pass

    @abstractmethod
    def embed(self, x):
        pass

    @abstractmethod
    def get_m(self):
        pass