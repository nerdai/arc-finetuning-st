from typing import List
from llama_index.core.bridge.pydantic import BaseModel, Field
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
    CORRECTION_PROMPT_TEMPLATE,
    Prediction,
    Critique,
    Correction,
)

example_template = """===
EXAMPLE

INPUT:
{input}

OUTPUT:
{output}
"""


class WorkflowOutput(BaseModel):
    passing: bool
    attempts: List[str]


class FinetuningDatasetWorkflow(Workflow):

    def __init__(self, llm: LLM, testing: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        self.testing = testing
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
    async def prediction(
        self, ctx: Context, ev: FormatTaskEvent
    ) -> PredictionEvent | StopEvent:
        prompt_vars = await ctx.get("prompt_vars")
        pred: Prediction = await self.llm.astructured_predict(
            Prediction, PREDICTION_PROMPT_TEMPLATE, **prompt_vars
        )
        attempts = [pred.prediction]
        await ctx.set("attempts", attempts)
        return PredictionEvent()

    @step
    async def evaluation(
        self, ctx: Context, ev: PredictionEvent | CorrectionEvent
    ) -> EvaluationEvent:
        task = await ctx.get("task")
        attempts = await ctx.get("attempts")
        prediction_str = attempts[-1]
        prediction = Prediction.prediction_str_to_int_array(prediction_str)
        ground_truth = task["test"][0]["output"]

        return EvaluationEvent(passing=(prediction == ground_truth))

    @step
    async def reflection(
        self, ctx: Context, ev: EvaluationEvent
    ) -> CorrectionEvent | StopEvent:
        attempts = await ctx.get("attempts")
        prompt_vars = await ctx.get("prompt_vars")
        prompt_vars.update(predicted_output=attempts[-1])  # use last attempt

        if ev.passing or (len(attempts) == self._max_attempts):
            result = WorkflowOutput(passing=ev.passing, attempts=attempts)
            return StopEvent(result=result)
        else:
            # generate critique
            critique_model: Critique = await self.llm.astructured_predict(
                Critique, REFLECTION_PROMPT_TEMPLATE, **prompt_vars
            )

            # work with human on critique

            # generate correction
            prompt_vars.update(critique=critique_model.critique)
            corr: Correction = await self.llm.astructured_predict(
                Correction, CORRECTION_PROMPT_TEMPLATE, **prompt_vars
            )
            if self.testing:
                return StopEvent(result=corr)
            attempts.append(corr.correction)
            await ctx.set("attempts", attempts)
            return CorrectionEvent()


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
