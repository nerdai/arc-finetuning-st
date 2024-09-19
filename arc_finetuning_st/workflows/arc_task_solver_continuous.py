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
    PredictionEvent,
    EvaluationEvent,
    CorrectionEvent,
)
from arc_finetuning_st.workflows.human_input import HumanInputWorkflow
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
    attempts: List[Prediction]


class ARCTaskSolverWorkflow(Workflow):

    def __init__(self, llm: LLM, max_attempts: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        self._max_attempts = max_attempts

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
        attempts = [pred]
        await ctx.set("attempts", attempts)
        return PredictionEvent()

    @step
    async def evaluation(
        self, ctx: Context, ev: PredictionEvent | CorrectionEvent
    ) -> EvaluationEvent:
        task = await ctx.get("task")
        attempts: List[Prediction] = await ctx.get("attempts")
        final_attempt = attempts[-1]
        prediction_str = final_attempt.prediction
        prediction = Prediction.prediction_str_to_int_array(prediction_str)
        ground_truth = task["test"][0]["output"]

        return EvaluationEvent(passing=(prediction == ground_truth))

    @step
    async def reflection(
        self, ctx: Context, ev: EvaluationEvent, human_input_workflow: Workflow
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

            human_prompt = (
                "\n\nA critique of the past incorrect prediction has been generated. "
                "\nCRITIQUE:\n\n"
                f"{critique_model.critique}"
                "\n\nIf you'd like to correct the critique, enter a new one now. "
                "Otherwise, return nothing.\n\n"
                "New critique:\n\n"
            )
            human_input = await human_input_workflow.run(
                prompt=human_prompt,
                critique=critique_model.critique,
                prediction_str=attempts[-1].prediction,
            )
            if human_input:
                critique_model.critique = human_input

            # generate correction
            prompt_vars.update(critique=critique_model.critique)
            corr: Correction = await self.llm.astructured_predict(
                Correction, CORRECTION_PROMPT_TEMPLATE, **prompt_vars
            )
            attempts.append(
                Prediction(
                    rationale=critique_model.critique, prediction=corr.correction
                )
            )
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

    w = ARCTaskSolverWorkflow(timeout=None, verbose=False, llm=OpenAI("gpt-4o"))
    w.add_workflows(human_input_workflow=HumanInputWorkflow())
    attempts = await w.run(task=task)

    print(attempts)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_workflow())
