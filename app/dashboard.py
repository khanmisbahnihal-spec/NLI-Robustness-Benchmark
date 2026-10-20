import streamlit as st

from utils import REPO_URL, inject_css

st.set_page_config(page_title="NLI Robustness Lab", layout="wide", initial_sidebar_state="expanded")
inject_css()

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/benchmark.py", title="Benchmark", icon=":material/bar_chart:"),
    st.Page("views/error_explorer.py", title="Error Explorer", icon=":material/search:"),
    st.Page("views/methodology.py", title="Methodology", icon=":material/menu_book:"),
]
st.sidebar.markdown(f"[GitHub repository]({REPO_URL})")
st.navigation(pages).run()
