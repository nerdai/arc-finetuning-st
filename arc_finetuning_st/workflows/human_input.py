from typing import Any, Awaitable, Protocol, runtime_checkable

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


@runtime_checkable
class HumanInputFn(Protocol):
    """Protocol for getting human input."""

    def __call__(self, prompt: str, **kwargs: Any) -> Awaitable[str]: ...


async def default_human_input_fn(prompt: str, **kwargs: Any) -> str:
    return input(prompt)


class HumanInputWorkflow(Workflow):
    def __init__(self, input: HumanInputFn = default_human_input_fn, **kwargs: Any):
        super().__init__(**kwargs)
        self.input = input

    @step
    async def human_input(self, ev: StartEvent) -> StopEvent:
        prompt = str(ev.get("prompt", ""))
        rationale = str(ev.get("rationale", ""))
        prediction_str = str(ev.get("prediction", ""))
        human_input = await self.input(
            prompt, rationale=rationale, prediction=prediction_str
        )
        return StopEvent(result=human_input)


# Local Testing
async def _test_workflow() -> None:
    w = HumanInputWorkflow(timeout=None, verbose=False)
    result = await w.run(prompt="How old are you?\n\n")
    print(str(result))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_workflow())
