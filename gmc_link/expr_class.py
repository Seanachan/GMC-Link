"""Three-way expression classifier for per-class specialist aligners (Tier B 1).

Splits Refer-KITTI expressions into {motion, static, appear} buckets used by
class-conditional training and inference routing. Priority motion > static >
appear. Mutually exclusive bucketing guarantees one specialist per expression.

Note: dataset.MOTION_KEYWORDS conflates active-motion (moving/turning) with
static qualifiers (parked/stopped). This module separates them so the static
specialist can train on parked-y exprs without active-motion contamination.
"""

# Active-motion kinematics: nonzero velocity learnable from 13D vector.
ACTIVE_MOTION_KEYWORDS = [
    "moving", "in motion", "driving",
    "walking", "running", "jogging", "crossing", "riding",
    "turning", "counter direction", "same direction", "opposite direction",
    "contrary direction", "reverse direction", "horizon direction",
    "heading", "in front of",
    "braking", "brake", "slower", "faster", "speedier",
    "accelerat", "decelerat", "slowing",
    "following", "approaching", "overtaking", "receding",
    "travelling", "traveling",
]

# Static qualifiers: zero or near-zero velocity. The 13D vector still encodes
# bbox state (cx, cy, w, h) so static specialist can learn position priors.
STATIC_KEYWORDS = [
    "parked", "parking", "at rest", "stationary", "still",
    "stopped", "stopping", "halted", "idle", "waiting", "standing",
]

# Appearance / person-attribute / spatial qualifiers (mirrors dataset.APPEARANCE_KEYWORDS).
APPEARANCE_KEYWORDS = [
    "red", "blue", "black", "white", "silver", "gray", "grey",
    "yellow", "green", "orange", "brown",
    "large", "small", "big", "tall", "short", "mini", "huge", "tiny",
    "suv", "sedan", "truck", "van", "hatchback", "wagon", "bus", "bike",
    "motorcycle", "bicycle",
    "left", "right", "near", "far", "front side", "behind",
    "man", "woman", "men", "women", "male", "female", "adult", "child",
    "kid", "girl", "boy", "person", "people",
    "wearing", "dressed", "jacket", "shirt", "pants", "hat", "bag",
    "shoes", "glasses", "coat",
]

CLASS_MOTION = "motion"
CLASS_STATIC = "static"
CLASS_APPEAR = "appear"
CLASS_LABELS = (CLASS_MOTION, CLASS_STATIC, CLASS_APPEAR)


def _has_any(text_lower, kw_list):
    return any(kw in text_lower for kw in kw_list)


def classify_expression(sentence):
    """Return one of {motion, static, appear}. Priority motion > static > appear.

    Fallback for expressions matching none of the three keyword sets: appear
    (most permissive bucket; fallback target for default routing too).
    """
    lower = sentence.lower()
    if _has_any(lower, ACTIVE_MOTION_KEYWORDS):
        return CLASS_MOTION
    if _has_any(lower, STATIC_KEYWORDS):
        return CLASS_STATIC
    return CLASS_APPEAR


def select_expressions_by_class(all_expressions, class_filter):
    """Filter expression list by class.

    class_filter: 'motion' | 'static' | 'appear' | 'all'.
    'all' = no filter (matches existing motion_filter='off' semantics).
    """
    if class_filter == "all":
        return list(all_expressions)
    if class_filter not in CLASS_LABELS:
        raise ValueError(
            f"class_filter must be 'motion'|'static'|'appear'|'all', got {class_filter!r}"
        )
    return [e for e in all_expressions if classify_expression(e["sentence"]) == class_filter]


def class_distribution(all_expressions):
    """Return dict of class → count for an expression list (for sanity logs)."""
    counts = {c: 0 for c in CLASS_LABELS}
    for e in all_expressions:
        counts[classify_expression(e["sentence"])] += 1
    return counts
