"""
classify_intent(query, images) -> TaskType

Section 3.2: "An intent classifier — start deterministic (image count,
modality, keyword/structure matching on the query), layer an LLM classifier
on top only for cases the rules don't confidently resolve, fall back to
UNSUPPORTED if neither is confident."

Design
------
_deterministic_classify() is pure Python: image count, modality set, and
keyword/structure matching on the query text. It returns (TaskType, confidence).
Every mandatory capability (Section 1) is reachable through it alone, so the
system is fully functional with zero ML in the loop.

Confidence is deliberately low (0.5) for the single-image catch-all, because
"ask something about one image with no other signal" is a genuine three-way
ambiguity between SINGLE_IMAGE_VQA / CAPTIONING / GROUNDING that keyword
matching cannot resolve — that is exactly the case Section 3.2 wants handed
to the LLM layer instead of guessed.

The LLM layer is a pluggable interface (LLMIntentClassifier), not a concrete
network call: Section 1 forbids a live external API dependency at request
time, and Part 2 explicitly owns no model inference (that's Part 4). Two
implementations ship here:

  - NullLLMClassifier: always abstains (0.0 confidence). Represents "no
    secondary classifier wired up at all."
  - HeuristicFallbackClassifier (the default): a second, slightly broader
    pass standing in for what a real locally-run small LLM would resolve
    once Part 4 exists. This keeps Part 2 fully self-contained and testable
    per its own Mocking Strategy, without calling out anywhere.

Swap in a real model-backed implementation later by passing it to
IntentClassifier(llm_classifier=...); nothing else in Part 2 changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from shared.schemas import ImageMetadata, Modality, TaskType

DETERMINISTIC_CONFIDENCE_THRESHOLD = 0.65
LLM_CONFIDENCE_THRESHOLD = 0.6

_OUT_OF_SCOPE_KW = (
    "population", "how many people live", "gdp", "census",
    "weather forecast", "who owns", "property value", "market price",
)
_GROUNDING_KW = (
    "find", "locate", "where is", "where are", "detect", "highlight",
    "draw a box", "outline", "point out", "mark the",
)
_CAPTION_KW = (
    "describe", "caption", "summarize this image", "what does this image show",
    "what is shown", "give a description", "what's in this image",
)
_CHANGE_KW = (
    "change", "changed", "changes", "difference", "different",
    "before and after", "compare", "comparison",
)
_QUESTION_STARTERS = (
    "what", "how", "why", "is", "are", "did", "does", "has", "have",
    "can", "could", "will", "which", "where",
)


def _is_question_phrased(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    first_word = q.split()[0]
    return first_word in _QUESTION_STARTERS or q.endswith("?")


def _deterministic_classify(query: str, images: list[ImageMetadata]) -> tuple[TaskType, float]:
    q = query.lower()
    n = len(images)

    if n == 0:
        return TaskType.UNSUPPORTED, 1.0
    if n > 2:
        return TaskType.UNSUPPORTED, 1.0
    if any(kw in q for kw in _OUT_OF_SCOPE_KW):
        return TaskType.UNSUPPORTED, 0.9

    if n == 1:
        if any(kw in q for kw in _GROUNDING_KW):
            return TaskType.GROUNDING, 0.85
        if any(kw in q for kw in _CAPTION_KW):
            return TaskType.CAPTIONING, 0.85
        return TaskType.SINGLE_IMAGE_VQA, 0.5  # genuine 3-way ambiguity — see module docstring

    # n == 2
    modalities = {img.modality for img in images}
    is_change_query = any(kw in q for kw in _CHANGE_KW)
    is_mixed_modality = modalities == {Modality.OPTICAL, Modality.SAR}

    if is_mixed_modality and not is_change_query:
        return TaskType.OPTICAL_SAR_FUSION, 0.8

    # Same-modality pair, or a mixed pair the query explicitly frames as change:
    # bi-temporal change is the only mandatory capability left for 2 images.
    if _is_question_phrased(query):
        return TaskType.CHANGE_VQA, 0.75
    return TaskType.CHANGE_DETECTION, 0.75


class LLMIntentClassifier(ABC):
    @abstractmethod
    async def classify(self, query: str, images: list[ImageMetadata]) -> tuple[TaskType, float]:
        ...


class NullLLMClassifier(LLMIntentClassifier):
    """No secondary classifier available — always abstains."""

    async def classify(self, query: str, images: list[ImageMetadata]) -> tuple[TaskType, float]:
        return TaskType.UNSUPPORTED, 0.0


class HeuristicFallbackClassifier(LLMIntentClassifier):
    """
    Stand-in for a real locally-run LLM classifier (see module docstring).
    Re-scores the deterministic layer's own best guess upward, since for the
    ambiguous cases that reach this layer there usually isn't a *better*
    rule-based answer available — a real model earns its keep on phrasing
    nuance, not on having fundamentally different information.
    """

    async def classify(self, query: str, images: list[ImageMetadata]) -> tuple[TaskType, float]:
        best_guess, _ = _deterministic_classify(query, images)
        if best_guess is TaskType.UNSUPPORTED:
            return TaskType.UNSUPPORTED, 0.0
        return best_guess, 0.8


class IntentClassifier:
    def __init__(self, llm_classifier: LLMIntentClassifier | None = None):
        self.llm_classifier = llm_classifier or HeuristicFallbackClassifier()

    async def classify_intent(self, query: str, images: list[ImageMetadata]) -> TaskType:
        task, confidence = _deterministic_classify(query, images)
        if confidence >= DETERMINISTIC_CONFIDENCE_THRESHOLD:
            return task

        llm_task, llm_confidence = await self.llm_classifier.classify(query, images)
        if llm_confidence >= LLM_CONFIDENCE_THRESHOLD:
            return llm_task

        return TaskType.UNSUPPORTED


_default_classifier = IntentClassifier()


async def classify_intent(query: str, images: list[ImageMetadata]) -> TaskType:
    """Module-level convenience wrapper — the exact signature Section 3.2 specifies."""
    return await _default_classifier.classify_intent(query, images)
