"""Factory for creating uncertainty quantification methods."""
from pathlib import Path
from typing import Dict, Optional, Type

from omegaconf import DictConfig, OmegaConf

from src.methods.method import Method



class MethodFactory:
    """Factory for creating method instances based on config."""

    _registry: Dict[str, Type[Method]] = {}

    @classmethod
    def register(cls, name: str, method_class: Type[Method]):
        """Register a method class with a name."""
        cls._registry[str(name).lower()] = method_class

    @classmethod
    def load_method_config(cls, config: DictConfig, method_name: Optional[str] = None) -> DictConfig:
        """Return a config with the requested method block loaded from config/method."""
        resolved_method_name = str(method_name or config.method.name).lower()
        config_copy = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        method_cfg = cls._find_method_config(resolved_method_name)

        if method_cfg is not None:
            config_copy.method = method_cfg
        else:
            config_copy.method.name = resolved_method_name

        return config_copy

    @classmethod
    def _find_method_config(cls, method_name: str) -> Optional[DictConfig]:
        method_dir = Path(__file__).resolve().parents[2] / "config" / "method"
        target = method_name.lower()

        for cfg_path in sorted(method_dir.glob("*.yaml")):
            if cfg_path.stem.lower() == target:
                return OmegaConf.load(cfg_path)

        for cfg_path in sorted(method_dir.glob("*.yaml")):
            loaded = OmegaConf.load(cfg_path)
            loaded_name = str(loaded.get("name", "")).lower()
            if loaded_name == target:
                return loaded

        return None

    @classmethod
    def create(cls, config: DictConfig, **kwargs) -> Method:
        """Create a method instance based on config.

        Args:
            config: Hydra configuration containing method specifications
            **kwargs: Additional keyword arguments to pass to method constructor

        Returns:
            Instantiated method

        Raises:
            ValueError: If method name not found in registry
        """
        config = cls.load_method_config(config)
        method_name = config.method.name.lower()

        if method_name not in cls._registry:
            raise ValueError(
                f"Unknown method: {method_name}. "
                f"Available methods: {list(cls._registry.keys())}"
            )

        return cls._registry[method_name](config, **kwargs)

    @classmethod
    def get_available_methods(cls) -> list:
        """Get list of available method names."""
        return list(cls._registry.keys())


def register_method(name: str):
    """Decorator to register a method class.
    
    Usage:
        @register_method("my_method")
        class MyMethod(Method):
            ...
    """
    def decorator(method_class: Type[Method]):
        MethodFactory.register(name, method_class)
        return method_class
    return decorator
