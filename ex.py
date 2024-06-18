import streamlit as st

st.set_page_config(
    page_title="Custom Themed App",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add your Streamlit app content here
st.title("Welcome to My Custom Themed Streamlit App")
st.write("This app uses a custom theme with 'Times New Roman' font.")