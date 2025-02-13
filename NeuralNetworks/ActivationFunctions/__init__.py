import pkgutil
import importlib
import inspect

from .ActivationFunction import ActivationFunction
from .ReLU import ReLU
from .HardSigmoid import HardSigmoid
from .HardTanh import HardTanh
from .HardSwish import HardSwish
from .HardShrink import HardShrink
from .Threshold import Threshold
from .Identity import Identity

AVAILABLE_ACTIVATION_FUNCTIONS = set()

package_name = __name__
package_path = __path__

for _, module_name, _ in pkgutil.iter_modules(package_path):
    module = importlib.import_module(f"{package_name}.{module_name}")

    # Inspect module attributes to find class definitions
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, ActivationFunction) and obj is not ActivationFunction:
            AVAILABLE_ACTIVATION_FUNCTIONS.add(obj)