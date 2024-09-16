from typing import List
from llama_index.core.workflow import (
    Event,
)


class FormatTaskEvent(Event): ...


class ReasoningEvent(Event):
    reasoning: str


class PredictionEvent(Event): ...


class EvaluationEvent(Event):
    passing: bool


class CorrectionEvent(Event): ...


class HumanInputEvent(Event):
    human_reasoning: str
