"""Python golden reference models for verifying generated Verilog modules.
All functions accept q and k as arguments — no hard-coded constants."""


def mod_mul(a: int, b: int, q: int, k: int | None = None) -> int:
    """(a * b * k) mod q if k is not None, else (a * b) mod q."""
    if k is not None:
        return (a * b * k) % q
    return (a * b) % q


def generate_test_vectors(n: int, q: int, seed: int = 42) -> list[dict]:
    """Generate boundary plus deterministic random vectors within [0, q-1]."""
    import random

    if n <= 0:
        return []
    boundary = [(0, 0), (0, q - 1), (1, q - 1), (q - 1, q - 1)]
    vectors = [{"a": a, "b": b} for a, b in boundary[:n]]
    if len(vectors) == n:
        return vectors
    rng = random.Random(seed)
    vectors.extend(
        {"a": rng.randint(0, q - 1), "b": rng.randint(0, q - 1)}
        for _ in range(n - len(vectors))
    )
    return vectors
