import streamlit as st

title = "Audio Sample Generator"

st.set_page_config(
    page_title=title,
)

body = ""

with open("README.md", "r") as file:
    lines = file.readlines()

    is_skipping_line = False

    for line in lines:
        is_skipping_line |= "installation: begin" in line
        is_skipping_line &= not "installation: end" in line

        if not is_skipping_line:
            body += line

st.markdown(
    body=body,
    unsafe_allow_html=True,
)

