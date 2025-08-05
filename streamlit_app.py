import streamlit as st
from app import home, analyze

PAGES = {
    "Home": home,
    "Analyze Report": analyze
}

st.set_page_config(page_title="Health Report Analyzer", layout="wide")

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
