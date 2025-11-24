# `PEFTExtender` - Parameter-Efficient Fine-Tuning Extension Base Class

## Class Signature

```python
class PEFTExtender(ABC):
    @abstractmethod
    def get_peft_params(self) -> Dict[str, Type]:
        ...

    @abstractmethod
    def apply_peft_config(self, ...) -> Dict[str, Any]:
        ...
```

**TBD**
