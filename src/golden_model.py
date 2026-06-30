"""Python golden reference models for verifying generated Verilog modules.
All functions accept q and k as arguments — no hard-coded constants."""


def mod_mul(a: int, b: int, q: int, k: int | None = None) -> int:
    """(a * b * k) mod q if k is not None, else (a * b) mod q."""
    if k is not None:
        return (a * b * k) % q
    return (a * b) % q


def mod_add(a: int, b: int, q: int) -> int:
    return (a + b) % q


def mod_sub(a: int, b: int, q: int) -> int:
    return (a - b) % q


def ct_butterfly(a: int, b: int, w: int, q: int, k: int | None = None) -> tuple[int, int]:
    """CT butterfly: a_out = a + b*w mod q, b_out = a - b*w mod q."""
    bw = mod_mul(b, w, q, k)
    return mod_add(a, bw, q), mod_sub(a, bw, q)


def generate_test_vectors(n: int, q: int, seed: int = 42) -> list[dict]:
    """Generate random test vectors within [0, q-1]."""
    import random
    rng = random.Random(seed)
    return [{"a": rng.randint(0, q - 1), "b": rng.randint(0, q - 1)} for _ in range(n)]
