from llama_index.core.workflow import (
    Workflow,
    Context,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.llms import LLM
from .events import ReasoningEvent, PredictionEvent, EvaluationEvent, HumanInputEvent


class FinetuningDatasetWorkflow(Workflow):

    def __init__(self, llm: LLM) -> None:
        self.llm = LLM

    @step
    async def reasoning(self, ev: StartEvent) -> ReasoningEvent:
        task = str(ev.get("task", {}))


async def _test_workflow():
    import json
    from pathlib import Path

    task_path = Path(
        Path(__file__).parents[2].absolute(), "data/training/0a938d79.json"
    )
    with open(task_path) as f:
        task = json.load(f)
    print(task)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_workflow())
