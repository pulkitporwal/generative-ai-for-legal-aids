import streamlit as st
import time
from main import generate_answer_with_rephrase

st.set_page_config(layout="wide")

chat_mode = st.sidebar.selectbox("Choose Option", options=["Get BNS Section Info", "FIR"])

st.title("Generative AI For Legal Aids")
st.divider()

prompt = st.chat_input("What is up?")

def typewriter_animation(text, speed=5):
    container = st.empty()  
    display_str = ""
    for char in text:
        display_str += char 
        container.markdown(f'<div class="msg-container"><p class="big-font">{display_str}</p></div>', unsafe_allow_html=True)
        time.sleep(1 / speed)  

if prompt:
    st.markdown("""
        <style>
        .big-font {
            font-size: 24px !important;
        }
        .msg-container{
            padding: 10px 14px !important;
            background-color : rgb(26, 28, 36) !important;
            border-radius: 10px !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    typewriter_animation("Welcome", speed=10)  
