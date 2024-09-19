import asyncio
import logging
import streamlit as st
import plotly.express as px
from typing import Any, Optional, List, Literal

from llama_index.llms.openai import OpenAI
from llama_deploy.control_plane import ControlPlaneConfig

from arc_finetuning_st.streamlit.examples import sample_tasks
from arc_finetuning_st.workflows.prompts import Prediction
from arc_finetuning_st.workflows.arc_task_solver import (
    ARCTaskSolverWorkflow,
    WorkflowOutput,
)
from arc_finetuning_st.workflows.human_input import HumanInputWorkflow, HumanInputFn

logger = logging.getLogger(__name__)


class Controller:
    def __init__(
        self, control_plane_config: Optional[ControlPlaneConfig] = None
    ) -> None:
        self.control_plane_config = control_plane_config or ControlPlaneConfig()

    def handle_selectbox_selection(self):
        """Handle selection of ARC task."""
        # clear prediction
        st.session_state.prediction = None
        st.session_state.disable_continue_button = True

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
        st.session_state.disable_continue_button = False

        critique = kwargs.get("critique", None)
        prediction_str = kwargs.get("prediction_str", None)
        grid = Prediction.prediction_str_to_int_array(prediction=prediction_str)
        fig = Controller.plot_grid(grid, kind="prediction")
        st.session_state.prediction = fig
        st.session_state.passing = False

        human_input = await asyncio.wait_for(Controller.get_human_input(), timeout=10)
        st.info(f"got human input: {human_input}")
        return human_input

    def handle_prediction_click(self) -> None:
        """Run workflow to generate prediction."""
        selected_task = st.session_state.selected_task
        task = sample_tasks.get(selected_task, None)
        if task:
            w = ARCTaskSolverWorkflow(
                timeout=None, verbose=False, llm=OpenAI("gpt-4o"), max_attempts=2
            )
            w.add_workflows(
                human_input_workflow=HumanInputWorkflow(input=Controller.human_input)
            )

            res: WorkflowOutput = asyncio.run(w.run(task=task))
            # final_attempt: Prediction = res.attempts[-1]
            # grid = Prediction.prediction_str_to_int_array(
            #     prediction=final_attempt.prediction
            # )
            # fig = Controller.plot_grid(grid, kind="prediction")
            # st.session_state.prediction = fig
            # st.session_state.critique = final_attempt.rationale
            # st.session_state.passing = res.passing
