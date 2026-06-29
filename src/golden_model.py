"""Python golden reference models for verifying generated Verilog modules.
All functions accept q and k as arguments — no hard-coded constants."""


def mod_mul(a: int, b: int, q: int, k: int | None = None) -> int:
    """
    Modular multiplication: (a * b * k) mod q  if k is not None,
                        or  (a * b) mod q      if k is None.
    """
    if k is not None:
        return (a * b * k) % q
    return (a * b) % q


def mod_add(a: int, b: int, q: int) -> int:
    return (a + b) % q


def mod_sub(a: int, b: int, q: int) -> int:
    return (a - b) % q


def ct_butterfly(a: int, b: int, w: int, q: int, k: int | None = None) -> tuple[int, int]:
    """
    Cooley-Tukey butterfly:  a_out = a + b*w mod q,  b_out = a - b*w mod q.
    If k is not None, w is assumed pre-multiplied by k^{-1}.
    """
    bw = mod_mul(b, w, q, k)
    return mod_add(a, bw, q), mod_sub(a, bw, q)


def generate_test_vectors(n: int, q: int, seed: int = 42) -> list[dict]:
    """Generate random test vectors within [0, q-1]."""
    import random
    rng = random.Random(seed)
    vectors = []
    for _ in range(n):
        a = rng.randint(0, q - 1)
        b = rng.randint(0, q - 1)
        vectors.append({"a": a, "b": b})
    return vectors


def parse_module_params(hardware_spec_params: dict) -> dict:
    """
    Extract numeric parameters from a HardwareSpec's parameters dict.
    Handles values like "12", "13 - Kyber constant k", "DATA_WIDTH - input bit width".
    Returns {"DATA_WIDTH": int, "Q": int, "K": int|None}.
    """
    import re

    def _extract_int(value: str) -> int | None:
        """Extract the first integer from a description string."""
        m = re.search(r'\d+', str(value))
        return int(m.group(0)) if m else None

    result: dict[str, int | None] = {
        "DATA_WIDTH": 12,
        "Q": 3329,
        "K": None,  # only set when the paper's spec explicitly defines K
    }

    for key, value in hardware_spec_params.items():
        v = _extract_int(value)
        if v is None:
            continue
        key_upper = key.upper()
        if key_upper in ("Q", "MODULUS", "MODULUS_Q", "MOD"):
            result["Q"] = v
        elif key_upper in ("K", "K_PARAM"):
            result["K"] = v
        elif key_upper in ("DATA_WIDTH", "WIDTH", "N_BITS", "DW"):
            result["DATA_WIDTH"] = v
        elif key_upper == "X":
            pass  # not needed for golden model

    return result
