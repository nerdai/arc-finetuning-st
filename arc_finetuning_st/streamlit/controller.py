import asyncio
import logging
import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Any, Dict, Optional, List, Literal

from llama_index.llms.openai import OpenAI

from arc_finetuning_st.streamlit.examples import sample_tasks
from arc_finetuning_st.workflows.prompts import Prediction
from arc_finetuning_st.workflows.arc_task_solver import (
    ARCTaskSolverWorkflow,
    WorkflowOutput,
)

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self) -> None:
        self._handler = None
        self._attempts = []
        self._passing_results = []

    def reset(self):
        # clear prediction
        st.session_state.prediction = None
        st.session_state.disable_continue_button = True
        st.session_state.disable_abort_button = True
        st.session_state.disable_start_button = False
        st.session_state.critique = None
        st.session_state.metric_value = "N/A"

        self._handler = None
        self._attempts = []
        self._passing_results = []

    def handle_selectbox_selection(self):
        """Handle selection of ARC task."""
        self.reset()

    @staticmethod
    def plot_grid(
        grid: List[List[int]], kind=Literal["input", "output", "prediction"]
    ) -> Any:
        m = len(grid)
        n = len(grid[0])
        fig = px.imshow(
            grid, text_auto=True, labels={"x": f"{kind.title()}<br><sup>{m}x{n}</sup>"}
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(
            yaxis={"visible": False},
            xaxis={"visible": True, "showticklabels": False},
            margin=dict(
                l=20,
                r=20,
                b=20,
                t=20,
            ),
        )
        return fig

    @staticmethod
    async def get_human_input() -> str:
        asyncio.sleep(3)
        return "got human input"

    @staticmethod
    async def human_input(prompt: str, **kwargs: Any) -> str:

        critique = kwargs.get("critique", None)
        prediction_str = kwargs.get("prediction_str", None)
        grid = Prediction.prediction_str_to_int_array(prediction=prediction_str)
        fig = Controller.plot_grid(grid, kind="prediction")
        st.session_state.prediction = fig
        st.session_state.passing = False

        human_input = await asyncio.wait_for(Controller.get_human_input(), timeout=10)
        st.info(f"got human input: {human_input}")
        return human_input

    def handle_abort_click(self) -> None:
        self.reset()

    async def handle_prediction_click(self) -> None:
        """Run workflow to generate prediction."""
        selected_task = st.session_state.selected_task
        task = sample_tasks.get(selected_task, None)
        if task:
            w = ARCTaskSolverWorkflow(timeout=None, verbose=False, llm=OpenAI("gpt-4o"))

            if not self._handler:  # start a new solver
                handler = w.run(task=task)
            else:  # continuing from past Workflow execution

                # use the critique and prediction str from streamlit
                critique = st.session_state.get("critique")
                prompt_vars = await self._handler.ctx.get("prompt_vars")
                prompt_vars.update(critique=critique)

                # check if selected rows
                selected_rows = (
                    st.session_state.get("attempts_history_df")
                    .get("selection")
                    .get("rows")
                )
                if selected_rows:
                    row_ix = selected_rows[0]
                    df_row = self.attempts_history_df.iloc[row_ix]
                    prediction_str = df_row["prediction"]
                    prompt_vars.update(predicted_output=prediction_str)

                await self._handler.ctx.set("prompt_vars", prompt_vars)

                # run Workflow
                handler = w.run(ctx=self._handler.ctx, task=task)

            res: WorkflowOutput = await handler
            self._handler = handler
            self._passing_results.append(res.passing)
            self._attempts = res.attempts

            # update streamlit states
            prompt_vars = await self._handler.ctx.get("prompt_vars")
            grid = Prediction.prediction_str_to_int_array(
                prediction=res.attempts[-1].prediction
            )
            prediction_fig = Controller.plot_grid(grid, kind="prediction")
            st.session_state.prediction = prediction_fig
            st.session_state.critique = prompt_vars["critique"]
            st.session_state.disable_continue_button = False
            st.session_state.disable_abort_button = False
            st.session_state.disable_start_button = True
            metric_value = "✅" if res.passing else "❌"
            st.session_state.metric_value = metric_value

    @property
    def passing(self) -> Optional[bool]:
        if self._passing_results:
            return self._passing_results[-1]
        return

    @property
    def attempts_history_df(
        self,
    ) -> pd.DataFrame:

        if self._attempts:
            attempt_number_list = []
            passings = []
            rationales = []
            predictions = []
            for ix, (a, passing) in enumerate(
                zip(self._attempts, self._passing_results)
            ):
                passings = ["✅" if passing else "❌"] + passings
                rationales = [a.rationale] + rationales
                predictions = [a.prediction] + predictions
                attempt_number_list = [ix + 1] + attempt_number_list
            return pd.DataFrame(
                {
                    "attempt #": attempt_number_list,
                    "passing": passings,
                    "rationale": rationales,
                    # hidden from UI
                    "prediction": predictions,
                }
            )
        return pd.DataFrame({})

    def handle_workflow_run_selection(self) -> None:
        selected_rows = (
            st.session_state.get("attempts_history_df").get("selection").get("rows")
        )
        if selected_rows:
            row_ix = selected_rows[0]
            df_row = self.attempts_history_df.iloc[row_ix]

            grid = Prediction.prediction_str_to_int_array(
                prediction=df_row["prediction"]
            )
            prediction_fig = Controller.plot_grid(grid, kind="prediction")
            st.session_state.prediction = prediction_fig
            st.session_state.critique = df_row["rationale"]
            metric_value = df_row["passing"]
            st.session_state.metric_value = metric_value
