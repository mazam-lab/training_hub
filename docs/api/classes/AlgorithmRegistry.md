# `AlgorithmRegistry` - Algorithm and Backend Registry

> Central registry for discovering and accessing available algorithms and their backends.

## Class Signature

```python
from training_hub import AlgorithmRegistry

class AlgorithmRegistry:
    """Registry for algorithms and their available backends."""

    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: Type[Algorithm]) -> None:
        """Register a new algorithm."""

    @classmethod
    def register_backend(cls, algorithm_name: str, backend_name: str, backend_class: Type[Backend]) -> None:
        """Register a backend for an algorithm."""

    @classmethod
    def get_algorithm(cls, name: str) -> Type[Algorithm]:
        """Get an algorithm class by name."""

    @classmethod
    def get_backend(cls, algorithm_name: str, backend_name: str) -> Type[Backend]:
        """Get a backend class for an algorithm."""

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """List all registered algorithms."""

    @classmethod
    def list_backends(cls, algorithm_name: str) -> list[str]:
        """List all backends for an algorithm."""
```

## Overview

`AlgorithmRegistry` is a centralized registry that manages all available algorithms and their backends in Training Hub. It provides a discovery mechanism for dynamically finding and instantiating algorithms and backends.

This is useful when:
- Building generic training frameworks
- Creating dynamic algorithm selection
- Discovering available capabilities at runtime
- Extending Training Hub with custom algorithms/backends

## Class Methods

### `register_algorithm(name: str, algorithm_class: Type[Algorithm]) -> None`

Registers a new algorithm in the registry.

**Parameters:**
- `name` (str): Unique name for the algorithm (e.g., `"sft"`, `"osft"`)
- `algorithm_class` (Type[Algorithm]): Algorithm class to register


**Example:**
```python
from training_hub import AlgorithmRegistry, Algorithm

class MyCustomAlgorithm(Algorithm):
    # ... implementation ...
    pass

AlgorithmRegistry.register_algorithm("my_algo", MyCustomAlgorithm)
```

### `register_backend(algorithm_name: str, backend_name: str, backend_class: Type[Backend]) -> None`

Registers a backend for a specific algorithm.
If a backend with the same name already exists for this algorithm, the old one will be overwritten.

**Parameters:**
- `algorithm_name` (str): Name of the algorithm this backend supports
- `backend_name` (str): Unique name for the backend
- `backend_class` (Type[Backend]): Backend class to register

**Raises:**
- `ValueError`: If the algorithm is not registered

**Example:**
```python
from training_hub import AlgorithmRegistry, Backend

class MyCustomBackend(Backend):
    # ... implementation ...
    pass

AlgorithmRegistry.register_backend("sft", "my_backend", MyCustomBackend)
```

### `get_algorithm(name: str) -> Type[Algorithm]`

Retrieves an algorithm class by name.

**Parameters:**
- `name` (str): Name of the algorithm

**Returns:**
- `Type[Algorithm]`: The algorithm class

**Raises:**
- `ValueError`: If the algorithm is not found

**Example:**
```python
from training_hub import AlgorithmRegistry

SFTClass = AlgorithmRegistry.get_algorithm("sft")
print(SFTClass)  # <class 'training_hub.algorithms.sft.SFTAlgorithm'>
```

### `get_backend(algorithm_name: str, backend_name: str) -> Type[Backend]`

Retrieves a backend class for a specific algorithm.

**Parameters:**
- `algorithm_name` (str): Name of the algorithm
- `backend_name` (str): Name of the backend

**Returns:**
- `Type[Backend]`: The backend class

**Raises:**
- `ValueError`: If the algorithm or backend is not found

**Example:**
```python
from training_hub import AlgorithmRegistry

BackendClass = AlgorithmRegistry.get_backend("sft", "instructlab-training")
print(BackendClass)  # <class '...InstructLabTrainingSFTBackend'>
```

### `list_algorithms() -> list[str]`

Returns a list of all registered algorithm names.

**Returns:**
- `list[str]`: List of algorithm names

**Example:**
```python
from training_hub import AlgorithmRegistry

algorithms = AlgorithmRegistry.list_algorithms()
print(algorithms)  # ['sft', 'osft']
```

### `list_backends(algorithm_name: str) -> list[str]`

Returns a list of all backends registered for a specific algorithm.

**Parameters:**
- `algorithm_name` (str): Name of the algorithm

**Returns:**
- `list[str]`: List of backend names

**Example:**
```python
from training_hub import AlgorithmRegistry

sft_backends = AlgorithmRegistry.list_backends("sft")
print(sft_backends)  # ['instructlab-training']

osft_backends = AlgorithmRegistry.list_backends("osft")
print(osft_backends)  # ['mini-trainer']
```

## Examples

### Discovering Available Algorithms

```python
from training_hub import AlgorithmRegistry

# List all algorithms
print("Available algorithms:")
for algo_name in AlgorithmRegistry.list_algorithms():
    backends = AlgorithmRegistry.list_backends(algo_name)
    print(f"  {algo_name}: backends={backends}")

# Output:
# Available algorithms:
#   sft: backends=['instructlab-training']
#   osft: backends=['mini-trainer']
```

### Dynamic Algorithm Selection

```python
from training_hub import AlgorithmRegistry, create_algorithm

def train_with_algorithm(algo_name: str, **kwargs):
    """Dynamically select and use an algorithm."""

    # Check if algorithm exists
    available = AlgorithmRegistry.list_algorithms()
    if algo_name not in available:
        raise ValueError(f"Unknown algorithm: {algo_name}. Available: {available}")

    # Get algorithm
    algorithm = create_algorithm(algo_name)

    # Train
    return algorithm.train(**kwargs)

# Use it
result = train_with_algorithm(
    "sft",
    model_path="./model",
    data_path="./data.jsonl",
    ckpt_output_dir="./output"
)
```

### Registering Custom Algorithm and Backend

```python
from training_hub import Algorithm, Backend, AlgorithmRegistry
from typing import Dict, Type, Any

# Define custom backend
class MyTrainingBackend(Backend):
    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        print(f"Training with custom backend: {algorithm_params}")
        # ... actual training implementation ...
        return {"status": "success"}

# Define custom algorithm
class MyFineTuningAlgorithm(Algorithm):
    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend
        self.kwargs = kwargs

    def train(self, **kwargs) -> Any:
        params = {**self.kwargs, **kwargs}
        return self.backend.execute_training(params)

    def get_required_params(self) -> Dict[str, Type]:
        return {
            "model_path": str,
            "data_path": str,
            "output_dir": str
        }

    def get_optional_params(self) -> Dict[str, Type]:
        return {
            "num_epochs": int,
            "batch_size": int
        }

# Register in the registry
AlgorithmRegistry.register_algorithm("myft", MyFineTuningAlgorithm)
AlgorithmRegistry.register_backend("myft", "my_backend", MyTrainingBackend)

# Verify registration
print(AlgorithmRegistry.list_algorithms())  # [..., 'myft']
print(AlgorithmRegistry.list_backends("myft"))  # ['my_backend']

# Use the new algorithm
from training_hub import create_algorithm

algo = create_algorithm("myft", backend_name="my_backend")
result = algo.train(
    model_path="./model",
    data_path="./data.jsonl",
    output_dir="./output"
)
```

### Building a Training Framework

```python
from training_hub import AlgorithmRegistry

class GenericTrainingFramework:
    """Framework that works with any registered algorithm."""

    def __init__(self):
        self.supported_algorithms = AlgorithmRegistry.list_algorithms()

    def get_algorithm_info(self, algo_name: str) -> dict:
        """Get information about an algorithm."""
        if algo_name not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        # Get algorithm class
        AlgoClass = AlgorithmRegistry.get_algorithm(algo_name)

        # Get available backends
        backends = AlgorithmRegistry.list_backends(algo_name)

        # Create temporary instance to inspect parameters
        from training_hub import create_algorithm
        temp_algo = create_algorithm(algo_name)

        return {
            "name": algo_name,
            "class": AlgoClass.__name__,
            "backends": backends,
            "required_params": temp_algo.get_required_params(),
            "optional_params": temp_algo.get_optional_params()
        }

    def list_capabilities(self):
        """List all training capabilities."""
        print("Training Framework Capabilities:")
        print(f"Supported algorithms: {self.supported_algorithms}")

        for algo_name in self.supported_algorithms:
            info = self.get_algorithm_info(algo_name)
            print(f"\n{algo_name}:")
            print(f"  Backends: {info['backends']}")
            print(f"  Required params: {list(info['required_params'].keys())}")

# Use the framework
framework = GenericTrainingFramework()
framework.list_capabilities()
```

### Checking Algorithm/Backend Availability

```python
from training_hub import AlgorithmRegistry

def is_algorithm_available(algo_name: str) -> bool:
    """Check if an algorithm is available."""
    return algo_name in AlgorithmRegistry.list_algorithms()

def is_backend_available(algo_name: str, backend_name: str) -> bool:
    """Check if a backend is available for an algorithm."""
    try:
        backends = AlgorithmRegistry.list_backends(algo_name)
        return backend_name in backends
    except ValueError:
        return False

# Use the checks
if is_algorithm_available("sft"):
    if is_backend_available("sft", "instructlab-training"):
        print("SFT with InstructLab backend is available!")
```

## Built-in Registrations

Training Hub automatically registers these algorithms and backends at import time:

**Algorithms:**
- `"sft"` → `SFTAlgorithm`
- `"osft"` → `OSFTAlgorithm`

**Backends:**
- `("sft", "instructlab-training")` → `InstructLabTrainingSFTBackend`
- `("osft", "mini-trainer")` → `MiniTrainerOSFTBackend`

## Implementation Notes

?> **Tip**: Use `list_algorithms()` and `list_backends()` to discover capabilities at runtime.

?> **Extensibility**: The registry pattern makes it easy to add custom algorithms and backends without modifying core code.


> **Note**: The registry is a class-level singleton. All registration is shared across your entire application.

> **Note**: Algorithms and backends are registered at module import time. Custom registrations should happen before using [`create_algorithm()`](/api/functions/create-algorithm).

## See Also

- [**create_algorithm() Function**](/api/functions/create-algorithm) - Factory function that uses this registry
- [**Algorithm Class**](/api/classes/Algorithm) - Base class for algorithms
- [**Backend Class**](/api/classes/Backend) - Base class for backends
- [**Extending the Framework Guide**](/guides/extending-framework) - Creating and registering custom algorithms

## Source

[View source on GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/__init__.py)
