from typing import List
from llama_index.core.workflow import (
    Event,
)


class ReasoningEvent(Event):
    reasoning: str


class PredictionEvent(Event):
    prediction: List[List[int]]


class EvaluationEvent(Event):
    correct: bool


class HumanInputEvent(Event):
    human_reasoning: str
