"""Run directly; a plugin can expose register through evotcr.factors entry points."""

import numpy as np
from evotcr.factors.registry import Factor, Registry


def register(registry):
    registry.register(
        Factor(
            name="example_contact_mean",
            version="1.0.0",
            family="example",
            definition="Row-local mean of supplied contact probabilities",
            dependencies=(),
            fitted=False,
            compute=lambda c: np.mean(c["contacts"], axis=1),
        )
    )


if __name__ == "__main__":
    r = Registry()
    register(r)
    print(r.build(["example_contact_mean"], {"contacts": np.array([[0.2, 0.8], [0.1, 0.3]])}))
