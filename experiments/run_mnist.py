import warnings

from src.experiments.aleatoric_trend_refactored import main


if __name__ == "__main__":
    warnings.warn(
        "`python -m experiments.run_mnist` has been retired. Use `python -m src.experiments.run_benchmark`.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
