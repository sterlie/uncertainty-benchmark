from typing import Dict, Type

from .model import Model


class ModelFactory:
    """Factory for creating model instances based on config."""

    _registry: Dict[str, Type[Model]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[Model]):
        cls._registry[name.lower()] = model_class

    @classmethod
    def create(cls, config) -> Model:
        model_name = config.model.name.lower()

        if model_name not in cls._registry:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(cls._registry.keys())}"
            )

        return cls._registry[model_name](config)

    @classmethod
    def available_models(cls) -> list:
        return list(cls._registry.keys())


def register_model(name: str):
    def decorator(model_class: Type[Model]):
        ModelFactory.register(name, model_class)
        return model_class

    return decorator
