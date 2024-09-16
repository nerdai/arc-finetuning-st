from typing import List
from llama_index.core.workflow import (
    Event,
)


class FormatTaskEvent(Event): ...


class ReasoningEvent(Event):
    reasoning: str


class PredictionEvent(Event): ...


class CorrectionEvent(Event):
    critique: str


class EvaluationEvent(Event):
    correct: bool


class HumanInputEvent(Event):
    human_reasoning: str
