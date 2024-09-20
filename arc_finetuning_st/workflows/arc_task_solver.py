from typing import Dict, List
from llama_index.core.bridge.pydantic import BaseModel
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
    PredictionEvent,
    EvaluationEvent,
)
from arc_finetuning_st.workflows.prompts import (
    REFLECTION_PROMPT_TEMPLATE,
    PREDICTION_PROMPT_TEMPLATE,
    CORRECTION_PROMPT_TEMPLATE,
)
from arc_finetuning_st.workflows.models import Prediction, Correction, Critique

example_template = """===
EXAMPLE

INPUT:
{input}

OUTPUT:
{output}
"""


class WorkflowOutput(BaseModel):
    passing: bool
    attempts: List[Prediction]


class ARCTaskSolverWorkflow(Workflow):

    def __init__(self, llm: LLM, max_attempts: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        self._max_attempts = max_attempts

    @step
    async def format_task(self, ctx: Context, ev: StartEvent) -> FormatTaskEvent:
        ctx.write_event_to_stream(ev)

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

        # check if ctx has data from previous run
        # if there is, don't overwrite it
        prompt_vars = await ctx.get("prompt_vars", {})

        if not prompt_vars:
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
        ctx.write_event_to_stream(ev)
        prompt_vars = await ctx.get("prompt_vars")

        if "critique" in prompt_vars:
            # generating a correction from last Workflow run
            attempts = await ctx.get("attempts")
            corr: Correction = await self.llm.astructured_predict(
                Correction, CORRECTION_PROMPT_TEMPLATE, **prompt_vars
            )
            attempts.append(
                Prediction(
                    rationale=prompt_vars["critique"], prediction=corr.correction
                )
            )
        else:
            # starting a new correction with no previous Workflow runs
            pred: Prediction = await self.llm.astructured_predict(
                Prediction, PREDICTION_PROMPT_TEMPLATE, **prompt_vars
            )
            attempts = [pred]

        await ctx.set("attempts", attempts)
        return PredictionEvent()

    @step
    async def evaluation(self, ctx: Context, ev: PredictionEvent) -> EvaluationEvent:
        ctx.write_event_to_stream(ev)
        task = await ctx.get("task")
        attempts: List[Prediction] = await ctx.get("attempts")
        final_attempt = attempts[-1]
        prediction_str = final_attempt.prediction
        prediction = Prediction.prediction_str_to_int_array(prediction_str)
        ground_truth = task["test"][0]["output"]

        return EvaluationEvent(passing=(prediction == ground_truth))

    @step
    async def reflection(self, ctx: Context, ev: EvaluationEvent) -> StopEvent:
        ctx.write_event_to_stream(ev)
        attempts: List[Prediction] = await ctx.get("attempts")

        # check if passing
        if not ev.passing:
            prompt_vars = await ctx.get("prompt_vars")
            prompt_vars.update(
                predicted_output=attempts[-1].prediction
            )  # use last attempt

            # generate critique
            critique_model: Critique = await self.llm.astructured_predict(
                Critique, REFLECTION_PROMPT_TEMPLATE, **prompt_vars
            )

            # generate correction
            prompt_vars.update(critique=critique_model.critique)
            await ctx.set("prompt_vars", prompt_vars)

        result = WorkflowOutput(passing=ev.passing, attempts=attempts)
        return StopEvent(result=result)


async def _test_workflow():
    import json
    from pathlib import Path
    from llama_index.llms.openai import OpenAI

    task_path = Path(
        Path(__file__).parents[2].absolute(), "data/training/0a938d79.json"
    )
    with open(task_path) as f:
        task = json.load(f)

    w = ARCTaskSolverWorkflow(timeout=None, verbose=False, llm=OpenAI("gpt-4o"))
    w.add_workflows(human_input_workflow=HumanInputWorkflow())
    attempts = await w.run(task=task)

    print(attempts)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_workflow())
