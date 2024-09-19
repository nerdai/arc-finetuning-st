import pandas as pd
import random
import streamlit as st
from typing import Tuple

from llama_index.core.tools.function_tool import async_to_sync

from arc_finetuning_st.streamlit.controller import Controller
from arc_finetuning_st.streamlit.examples import sample_tasks

st.set_page_config(layout="wide")


@st.cache_resource
def startup() -> Tuple[Controller,]:
    controller = Controller()
    return (controller,)


(controller,) = startup()


if "passing" not in st.session_state:
    st.session_state["passing"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = ""
if "disable_continue_button" not in st.session_state:
    st.session_state["disable_continue_button"] = True
if "attempts" not in st.session_state:
    st.session_state["attempts"] = {}

logo = '[<img src="https://d3ddy8balm3goa.cloudfront.net/llamaindex/LlamaLogoSquare.png" width="28" height="28" />](https://github.com/run-llama/llama-agents "Check out the llama-agents Github repo!")'
st.title("ARC Task Solver Workflow with Human Input")


with st.sidebar:
    task_selection = st.selectbox(
        label="Task",
        options=sample_tasks.keys(),
        placeholder="Select a task.",
        index=None,
        on_change=controller.handle_selectbox_selection,
        key="selected_task",
    )

train_col, test_col = st.columns([1, 1], vertical_alignment="top", gap="medium")

with train_col:
    st.subheader("Train Examples")
    with st.container():
        selected_task = st.session_state.selected_task
        task = sample_tasks.get(selected_task, None)

        if task:
            num_examples = len(task["train"])
            tabs = st.tabs([f"Example {ix}" for ix in range(1, num_examples + 1)])
            for ix, tab in enumerate(tabs):
                with tab:
                    left, right = st.columns(
                        [1, 1], vertical_alignment="top", gap="medium"
                    )
                    with left:
                        ex = task["train"][ix]
                        grid = ex["input"]
                        fig = Controller.plot_grid(grid, kind="input")
                        st.plotly_chart(fig, use_container_width=True)

                    with right:
                        ex = task["train"][ix]
                        grid = ex["output"]
                        fig = Controller.plot_grid(grid, kind="output")
                        st.plotly_chart(fig, use_container_width=True)

with test_col:
    header_col, start_col = st.columns([5, 1], vertical_alignment="bottom", gap="small")
    with header_col:
        st.subheader("Test")
    with start_col:
        st.button(
            "start",
            on_click=async_to_sync(controller.handle_prediction_click),
            use_container_width=True,
        )
    with st.container():
        selected_task = st.session_state.selected_task
        task = sample_tasks.get(selected_task, None)

        if task:
            num_cases = len(task["test"])
            tabs = st.tabs([f"Test Case {ix}" for ix in range(1, num_cases + 1)])
            for ix, tab in enumerate(tabs):
                with tab:
                    left, right = st.columns(
                        [1, 1], vertical_alignment="top", gap="medium"
                    )
                    with left:
                        ex = task["test"][ix]
                        grid = ex["input"]
                        fig = Controller.plot_grid(grid, kind="input")
                        st.plotly_chart(fig, use_container_width=True)

                    with right:
                        prediction_fig = st.session_state.get("prediction", None)
                        if prediction_fig:
                            st.plotly_chart(
                                prediction_fig,
                                use_container_width=True,
                                key="prediction",
                            )

        with st.container():
            # metric
            passing = st.session_state.get("passing")
            if passing is None:
                metric_value = "N/A"
            elif passing == True:
                metric_value = "✅"
            else:
                metric_value = "❌"

            st.metric(label="Passing", value=metric_value)

            # console
            st.text_area(label="human input", key="critique")

            st.button(
                "continue",
                on_click=controller.handle_prediction_click,
                use_container_width=True,
                disabled=st.session_state.get("disable_continue_button"),
                key="continue_button",
            )

            df = pd.DataFrame(
                {
                    "name": ["Roadmap", "Extras", "Issues"],
                    "url": [
                        "https://roadmap.streamlit.app",
                        "https://extras.streamlit.app",
                        "https://issues.streamlit.app",
                    ],
                    "stars": [random.randint(0, 1000) for _ in range(3)],
                    "views_history": [
                        [random.randint(0, 5000) for _ in range(30)] for _ in range(3)
                    ],
                }
            )
            st.dataframe(
                df,
                column_config={
                    "name": "App name",
                    "stars": st.column_config.NumberColumn(
                        "Github Stars",
                        help="Number of stars on GitHub",
                        format="%d ⭐",
                    ),
                    "url": st.column_config.LinkColumn("App URL"),
                    "views_history": st.column_config.LineChartColumn(
                        "Views (past 30 days)", y_min=0, y_max=5000
                    ),
                },
                hide_index=True,
            )
