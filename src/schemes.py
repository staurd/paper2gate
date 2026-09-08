"""PQC scheme profiles: fixed arithmetic constants and pipeline integration flags.

Each scheme's constants (modulus q, operand width, product width) are
cryptographic ground truth — they must never be inferred from an LLM
extraction. The profile drives prompt rendering, golden-model verification
parameters, and Stage 3b reference-file selection.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SchemeProfile:
    name: str
    q: int
    data_width: int            # operand width in bits (ceil(log2(q)))
    product_width: int         # P_DSP / P_R width (2 * data_width)
    default_latency: int       # verification TB latency param (see verification_runner)
    reference_files: tuple[str, ...]  # reference/ files for Stage 3b run 2
    integrate: bool            # Stage 3 butterfly integration supported


KYBER = SchemeProfile(
    "kyber",
    q=3329,
    data_width=12,
    product_width=24,
    default_latency=3,
    reference_files=("butterfly.v", "div2.v", "modadd.v", "modsub.v"),
    integrate=True,
)

DILITHIUM = SchemeProfile(
    "dilithium",
    q=8380417,
    data_width=23,
    product_width=46,
    default_latency=5,
    reference_files=(),
    integrate=False,
)

SCHEMES = {"kyber": KYBER, "dilithium": DILITHIUM}


def detect_scheme(paper_text: str) -> SchemeProfile:
    """Detect the PQC scheme from the paper text.

    Constants are checked over the FULL text and take priority over keywords
    (they may appear deep in the body, not in the title/abstract). 8380417 is
    checked before 3329: Dilithium papers routinely cite Kyber's modulus, but
    Kyber papers never contain Dilithium's. Keywords are only a fallback for
    papers that discuss a scheme without printing its modulus.
    """
    norm = re.sub(r"[,\s]", "", paper_text)  # tolerate "8,380,417" formatting
    if "8380417" in norm:
        return DILITHIUM
    if "3329" in norm:
        return KYBER

    head = paper_text[:2000].lower()
    if "dilithium" in head:
        return DILITHIUM
    if "kyber" in head or "crystals" in head:
        return KYBER

    # Fallback: KYBER is the project's original default; the spec-conflict
    # guard in main.py warns if a Dilithium paper slips through here.
    return KYBER


def resolve_scheme(scheme: str | None, paper_text: str) -> SchemeProfile:
    """Resolve a --scheme CLI value (None/"auto" = detect) to a profile."""
    if scheme in (None, "auto"):
        return detect_scheme(paper_text)
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
