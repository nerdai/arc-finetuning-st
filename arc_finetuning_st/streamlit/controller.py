import logging
import streamlit as st
import plotly.express as px
from typing import Any, Optional, List, Literal

from llama_deploy.control_plane import ControlPlaneConfig

from arc_finetuning_st.streamlit.examples import sample_tasks

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

    def handle_prediction_click(self) -> None:
        """Run workflow to generate prediction."""
        selected_task = st.session_state.selected_task
        task = sample_tasks.get(selected_task, None)
        if task:
            grid = task["test"][0]["output"]
            fig = Controller.plot_grid(grid, kind="prediction")
            st.session_state.prediction = fig
