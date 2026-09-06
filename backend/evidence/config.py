"""
Tunable thresholds for Part 5 (Evidence Validation, Confidence & Response
Generation).

Every threshold here is a judgment call the architecture doc leaves to the
implementer. Nothing else in this package hardcodes a magic number outside
this file — change behavior by editing these constants, not by hunting
through validator.py / confidence.py / claim_check.py.
"""

# --- Confidence banding ----------------------------------------------------
# A computed confidence.value >= this -> "HIGH"; >= MEDIUM threshold -> band
# "MEDIUM"; anything else (including None) -> "LOW".
CONFIDENCE_HIGH_THRESHOLD = 0.75
CONFIDENCE_MEDIUM_THRESHOLD = 0.40

# --- Abstain rules -----------------------------------------------------------
# Evidence counts as "weak" (validator.py should abstain) when a calibrated
# confidence value is present but falls below this floor, AND no independent
# stronger signal (a decent detection score / change-map confidence) exists
# to lean on instead.
WEAK_EVIDENCE_FLOOR = 0.25

# Substring (case-insensitive) that marks a warning as a co-registration
# refusal per Part 3's contract. Part 2 already owns refusing the downstream
# task when `CoregistrationResult.aligned` is False (see Part 2 hardening),
# but Part 5 checks again in case that call path ever changes.
COREGISTRATION_WARNING_MARKER = "coregistration"

# --- Numeric claim-check tolerance -----------------------------------------
# Two numbers are treated as "the same fact" if they match after rounding to
# this many decimal places, OR are within this relative tolerance of one
# another. This exists so ordinary natural-language rounding ("about 12%"
# for a true value of 12.4) doesn't get flagged as fabrication, while a
# genuinely invented number (87% for a true value of 12.4) still is. 5% was
# chosen as generous enough for that kind of rounding, tight enough that it
# won't wave through a materially different number.
CLAIM_CHECK_DECIMAL_PLACES = 1
CLAIM_CHECK_RELATIVE_TOLERANCE = 0.05  # 5%

# --- Generation retry policy -------------------------------------------------
# How many extra times to ask the LLM again, with a stricter prompt, after a
# numeric claim-check failure — before giving up and using the fully
# templated, non-LLM sentence (which is guaranteed to pass by construction).
MAX_LLM_REGENERATION_ATTEMPTS = 1
