from abc import ABC, abstractmethod
class KernelFunctionBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_kernel(self):
        pass 

    @abstractmethod
    def get_kernel_diag(self):
        pass

    @abstractmethod
    def kernel(self, a,b, **kwargs):
        pass

    @abstractmethod
    def kernel_diag(self, a,b, **kwargs):
        pass
