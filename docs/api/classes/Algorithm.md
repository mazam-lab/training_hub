# `Algorithm` - Base Class for Training Algorithms

> Abstract base class that defines the interface for all training algorithms in Training Hub.

## Class Signature

```python
from training_hub import Algorithm
from abc import ABC, abstractmethod

class Algorithm(ABC):
    """Base class for all training algorithms."""

    @abstractmethod
    def train(self, **kwargs) -> Any:
        """Execute the training algorithm."""
        pass

    @abstractmethod
    def get_required_params(self) -> Dict[str, Type]:
        """Return dictionary of required parameter names and their types."""
        pass

    @abstractmethod
    def get_optional_params(self) -> Dict[str, Type]:
        """Return dictionary of optional parameter names and their types."""
        pass
```

## Overview

The `Algorithm` class provides a common interface for all training algorithms in Training Hub. It uses Python's Abstract Base Class (ABC) pattern to ensure that all concrete algorithm implementations provide the necessary methods.

This class is primarily useful for:
- Understanding the Training Hub architecture
- Creating custom algorithms
- Type hinting in generic training functions

## Abstract Methods

All classes inheriting from `Algorithm` **must** implement these methods:

### `train(**kwargs) -> Any`

Executes the training algorithm with the given parameters.

**Parameters:**
- `**kwargs`: Algorithm-specific training parameters

**Returns:**
- `Any`: Training result (structure depends on the implementation)

**Raises:**
- `ValueError`: If required parameters are missing or invalid

### `get_required_params() -> Dict[str, Type]`

Returns a dictionary mapping required parameter names to their Python types.

**Returns:**
- `Dict[str, Type]`: Dictionary with parameter names as keys and types as values

**Example return value:**
```python
{
    "model_path": str,
    "data_path": str,
    "ckpt_output_dir": str
}
```

### `get_optional_params() -> Dict[str, Type]`

Returns a dictionary mapping optional parameter names to their Python types.

**Returns:**
- `Dict[str, Type]`: Dictionary with parameter names as keys and types as values

**Example return value:**
```python
{
    "num_epochs": int,
    "learning_rate": float,
    "effective_batch_size": int
}
```

## Concrete Implementations

Training Hub provides these concrete `Algorithm` implementations:

| Class | Algorithm | Description |
|-------|-----------|-------------|
| [`SFTAlgorithm`](/api/classes/SFTAlgorithm) | Supervised Fine-Tuning | Standard supervised fine-tuning |
| [`OSFTAlgorithm`](/api/classes/OSFTAlgorithm) | Orthogonal Subspace Fine-Tuning | Continual learning without catastrophic forgetting |

## Examples

### Using an Existing Algorithm

```python
from training_hub import SFTAlgorithm, InstructLabTrainingSFTBackend

# Create backend
backend = InstructLabTrainingSFTBackend()

# Create algorithm instance
algorithm = SFTAlgorithm(backend=backend)

# Check what parameters are available
print("Required:", algorithm.get_required_params())
print("Optional:", algorithm.get_optional_params())

# Train
result = algorithm.train(
    model_path="./model",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints"
)
```

### Creating a Custom Algorithm

```python
from training_hub import Algorithm, Backend
from typing import Dict, Type, Any

class MyCustomAlgorithm(Algorithm):
    """Custom training algorithm."""

    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend
        self.kwargs = kwargs

    def train(self, **kwargs) -> Any:
        """Execute custom training logic."""
        # Validate parameters
        required = self.get_required_params()
        for param, param_type in required.items():
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")

        # Prepare parameters for backend
        algorithm_params = {**self.kwargs, **kwargs}

        # Execute training via backend
        return self.backend.execute_training(algorithm_params)

    def get_required_params(self) -> Dict[str, Type]:
        """Define required parameters."""
        return {
            "model_path": str,
            "data_path": str,
            "output_dir": str
        }

    def get_optional_params(self) -> Dict[str, Type]:
        """Define optional parameters."""
        return {
            "num_steps": int,
            "batch_size": int,
            "learning_rate": float
        }

# Register your algorithm
from training_hub import AlgorithmRegistry

AlgorithmRegistry.register_algorithm("my_algo", MyCustomAlgorithm)
```

### Generic Training Function Using Algorithm Interface

```python
from training_hub import Algorithm

def validate_and_train(algorithm: Algorithm, **params) -> Any:
    """Validate parameters and execute training."""

    # Get algorithm parameter specifications
    required = algorithm.get_required_params()
    optional = algorithm.get_optional_params()

    # Validate required parameters
    missing = []
    for param in required:
        if param not in params:
            missing.append(param)

    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

    # Warn about unknown parameters
    known_params = set(required.keys()) | set(optional.keys())
    unknown = set(params.keys()) - known_params

    if unknown:
        print(f"Warning: Unknown parameters will be ignored: {unknown}")

    # Execute training
    return algorithm.train(**params)

# Use it
from training_hub import create_algorithm

sft_algo = create_algorithm("sft")
result = validate_and_train(
    sft_algo,
    model_path="./model",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints"
)
```

## Design Pattern

The `Algorithm` class follows the **Strategy Pattern**, where:
- `Algorithm` defines the interface for training strategies
- Concrete implementations (`SFTAlgorithm`, `OSFTAlgorithm`) provide specific algorithms
- `Backend` classes handle the actual training execution
- Client code can switch between algorithms without changing its structure

## Implementation Notes

?> **For Users**: You typically don't need to interact with this class directly. Use the convenience functions ([`sft()`](/api/functions/sft), [`osft()`](/api/functions/osft)) or concrete algorithm classes instead.

?> **For Framework Developers**: When creating a custom algorithm, inherit from this class and implement all three abstract methods. See the [Extending the Framework Guide](/guides/extending-framework) for details.

> **Note**: All algorithm instances require a [`Backend`](/api/classes/Backend) instance to execute the actual training. The backend handles framework-specific implementation details.

## See Also

- [**Backend Class**](/api/classes/Backend) - Base class for backend implementations
- [**SFTAlgorithm Class**](/api/classes/SFTAlgorithm) - Concrete SFT implementation
- [**OSFTAlgorithm Class**](/api/classes/OSFTAlgorithm) - Concrete OSFT implementation
- [**AlgorithmRegistry Class**](/api/classes/AlgorithmRegistry) - Registry for algorithms and backends
- [**Extending the Framework Guide**](/guides/extending-framework) - Creating custom algorithms

## Source

[View source on GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/__init__.py)
