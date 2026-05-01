from src.experiments.datasets.registry import _ADAPTERS


def get_available_datasets() -> list[str]:
    """Return the names of all registered datasets."""
    return sorted(_ADAPTERS.keys())


if __name__ == "__main__":
    print("Available datasets:", get_available_datasets())