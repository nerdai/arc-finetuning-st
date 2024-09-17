import streamlit as st

st.set_page_config(layout="wide")

logo = '[<img src="https://d3ddy8balm3goa.cloudfront.net/llamaindex/LlamaLogoSquare.png" width="28" height="28" />](https://github.com/run-llama/llama-agents "Check out the llama-agents Github repo!")'
st.title("Human In The Loop Workflows: ARC Solver")
st.markdown(f"_Powered by LlamaIndex_ &nbsp; {logo}", unsafe_allow_html=True)
