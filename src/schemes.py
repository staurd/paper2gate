"""PQC scheme profiles: fixed arithmetic constants for modmul generation.

Each scheme's constants (modulus q, operand width, product width) are
cryptographic ground truth — they must never be inferred from an LLM
extraction. The profile drives prompt rendering and golden-model verification.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SchemeProfile:
    name: str
    q: int
    data_width: int            # operand width in bits (ceil(log2(q)))
    product_width: int         # P_DSP / P_R width (2 * data_width)
    default_latency: int       # verification TB latency param (see verification_runner)


KYBER = SchemeProfile(
    "kyber",
    q=3329,
    data_width=12,
    product_width=24,
    default_latency=3,
)

DILITHIUM = SchemeProfile(
    "dilithium",
    q=8380417,
    data_width=23,
    product_width=46,
    default_latency=5,
)

SCHEMES = {"kyber": KYBER, "dilithium": DILITHIUM}


def resolve_scheme(scheme: str) -> SchemeProfile:
    """Return the fixed arithmetic profile for an explicitly selected scheme."""
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose from: {sorted(SCHEMES)}")
    return SCHEMES[scheme]


def render_vars(p: SchemeProfile) -> dict:
    """Jinja variables for extract_innovation.jinja and generate_modmul.jinja."""
    return {
        "scheme_name": p.name,
        "q": p.q,
        "q_minus_1": p.q - 1,
        "dw": p.data_width,
        "dw_minus_1": p.data_width - 1,
        "dw_plus_1": p.data_width + 1,
        "pw": p.product_width,
        "pw_minus_1": p.product_width - 1,
        "q_minus_1_sq": (p.q - 1) ** 2,
    }
