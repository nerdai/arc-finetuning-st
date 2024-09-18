import plotly.express as px
import streamlit as st
from typing import Tuple

from arc_finetuning_st.streamlit.controller import Controller
from arc_finetuning_st.streamlit.examples import sample_tasks

st.set_page_config(layout="wide")


@st.cache_resource
def startup() -> Tuple[Controller,]:
    controller = Controller()
    return (controller,)


(controller,) = startup()


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
                        fig = Controller.plot_grid(grid, kind="input")
                        st.plotly_chart(fig, use_container_width=True)

with test_col:
    st.subheader("Test")
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
        critique = st.session_state.get("critique", None)
        st.text_area(
            value=critique,
            label="Critique",
            placeholder="Enter critique.",
            key="critique",
        )
        st.button(
            "predict", type="primary", on_click=controller.handle_prediction_click
        )
