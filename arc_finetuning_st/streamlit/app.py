import streamlit as st

st.set_page_config(layout="wide")

logo = '[<img src="https://d3ddy8balm3goa.cloudfront.net/llamaindex/LlamaLogoSquare.png" width="28" height="28" />](https://github.com/run-llama/llama-agents "Check out the llama-agents Github repo!")'
st.title("ARC Task Solver Workflow with Human Input")
st.markdown(f"_Powered by LlamaIndex_ &nbsp; {logo}", unsafe_allow_html=True)

with st.container():
    task_selection = st.selectbox(
        label="Task",
        options=["1.json", "2.json", "3.json"],
        placeholder="Select a task.",
        index=None,
    )

train_col, test_col = st.columns([1, 1], vertical_alignment="top", gap="medium")

with test_col:
    with st.container(border=True):
        left, right = st.columns([1, 1], vertical_alignment="top", gap="medium")
        with left:
            st.subheader("Input")
            _ = st.container(border=True, height=300)

        with right:
            st.subheader("Output Prediction")
            _ = st.container(border=True, height=300)
        st.text_area(label="Critique", placeholder="Enter critique.")
        st.button("predict", type="primary")

with train_col:
    st.subheader("Train Examples")
    with st.container(border=True):
        left, right = st.columns([1, 1], vertical_alignment="top", gap="medium")
        with left:
            st.subheader("Input")
            _ = st.container(border=True, height=300)

        with right:
            st.subheader("Output Prediction")
            _ = st.container(border=True, height=300)
