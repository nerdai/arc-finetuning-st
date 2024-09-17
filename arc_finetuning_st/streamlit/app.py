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
st.markdown(f"_Powered by LlamaIndex_ &nbsp; {logo}", unsafe_allow_html=True)

with st.container():
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
    with st.container(border=True, height=800):
        left, right = st.columns([1, 1], vertical_alignment="top", gap="medium")
        with left:
            st.subheader("Input")
            selected_task = st.session_state.selected_task
            task = sample_tasks.get(selected_task, None)
            if task:
                for ix, ex in enumerate(task["train"]):
                    grid = ex["input"]
                    fig = px.imshow(grid, text_auto=True)
                    fig.update_coloraxes(showscale=False)
                    fig.update_layout(
                        yaxis={"visible": False}, xaxis={"visible": False}
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Output")
            selected_task = st.session_state.selected_task
            task = sample_tasks.get(selected_task, None)
            if task:
                for ex in task["train"]:
                    grid = ex["output"]
                    fig = px.imshow(grid, text_auto=True)
                    fig.update_coloraxes(showscale=False)
                    fig.update_layout(
                        yaxis={"visible": False}, xaxis={"visible": False}
                    )
                    st.plotly_chart(fig, use_container_width=True)

with test_col:
    st.subheader("Test Example")
    with st.container(border=True):
        left, right = st.columns([1, 1], vertical_alignment="top", gap="medium")
        with left:
            st.subheader("Input")
            selected_task = st.session_state.selected_task
            task = sample_tasks.get(selected_task, None)
            if task:
                grid = task["test"][0]["input"]
                fig = px.imshow(grid, text_auto=True)
                fig.update_coloraxes(showscale=False)
                fig.update_layout(yaxis={"visible": False}, xaxis={"visible": False})
                st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Output Prediction")
            _ = st.container(border=True, height=300)
        st.text_area(label="Critique", placeholder="Enter critique.")
        st.button("predict", type="primary")
