from abc import ABC, abstractmethod
from z3 import BoolRef, Real


class ActivationFunction(ABC):

    def __call__(self, x: float) -> float:
        return self.apply(x)

    def __str__(self):
        attributes = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attributes})"

    @abstractmethod
    def apply(self, x: float) -> float:
        """
        Applies the activation function to the given input.
        """
        ...

    @abstractmethod
    def formula(self, x: Real) -> BoolRef:
        """
        Returns the z3 SMT formula that represents the input-output relation of this activation function.
        
        :param x: the z3 variable that represents the input to the activation function
        """
        ...
