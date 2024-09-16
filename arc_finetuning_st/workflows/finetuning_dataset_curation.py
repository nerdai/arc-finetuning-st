import json

from llama_index.core.workflow import (
    Workflow,
    Context,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.llms import LLM
from arc_finetuning_st.workflows.events import (
    FormatTaskEvent,
    ReasoningEvent,
    PredictionEvent,
    EvaluationEvent,
    HumanInputEvent,
    CorrectionEvent,
)
from arc_finetuning_st.workflows.prompts import (
    REFLECTION_PROMPT_TEMPLATE,
    PREDICTION_PROMPT_TEMPLATE,
)

example_template = """===
EXAMPLE

INPUT:
{input}

OUTPUT:
{output}
"""


class FinetuningDatasetWorkflow(Workflow):

    def __init__(self, llm: LLM, **kwargs) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        self._max_attempts = 3

    @step
    async def format_task(self, ctx: Context, ev: StartEvent) -> FormatTaskEvent:
        from typing import Dict, List

        def _format_row(row: List[int]) -> str:
            return ",".join(str(el) for el in row)

        def pretty_print_grid(grid: List[List[int]]) -> str:
            formatted_rows = [_format_row(row) for row in grid]
            return "\n".join(formatted_rows)

        def format_train_example(train_pair: Dict) -> str:
            return example_template.format(
                input=pretty_print_grid(train_pair["input"]),
                output=pretty_print_grid(train_pair["output"]),
            )

        task = ev.get("task", {})
        await ctx.set("task", task)
        examples = [format_train_example(t) for t in task["train"]]
        prompt_vars = {
            "test_input": pretty_print_grid(task["test"][0]["input"]),
            "examples": "\n".join(examples),
        }
        await ctx.set("prompt_vars", prompt_vars)
        return FormatTaskEvent()

    @step
    async def prediction(self, ctx: Context, ev: FormatTaskEvent) -> PredictionEvent:
        prompt_vars = await ctx.get("prompt_vars")
        pred = await self.llm.apredict(PREDICTION_PROMPT_TEMPLATE, **prompt_vars)
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            raise
        attempts = [pred["output"]]
        await ctx.set("attempts", attempts)
        return PredictionEvent()

    @step
    async def reflection(
        self, ctx: Context, ev: PredictionEvent | CorrectionEvent
    ) -> CorrectionEvent | StopEvent:
        attempts = await ctx.get("attempts")
        prompt_vars = await ctx.get("prompt_vars")
        prompt_vars.update(predicted_output=attempts[-1])  # use last attempt

        if len(attempts) == self._max_attempts:
            return StopEvent(result=attempts)
        else:
            # generate critique
            corr = await self.llm.apredict(REFLECTION_PROMPT_TEMPLATE, **prompt_vars)
            try:
                corr = json.loads(corr)
            except json.JSONDecodeError:
                raise
            attempts.append(corr["corrected_output"])
            await ctx.set("attempts", attempts)
            return CorrectionEvent(critique=corr["critique"])


async def _test_workflow():
    import json
    from pathlib import Path
    from llama_index.llms.openai import OpenAI

    task_path = Path(
        Path(__file__).parents[2].absolute(), "data/training/0a938d79.json"
    )
    with open(task_path) as f:
        task = json.load(f)

    w = FinetuningDatasetWorkflow(timeout=None, verbose=False, llm=OpenAI("gpt-4o"))
    attempts = await w.run(task=task)

    print(attempts)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_workflow())
