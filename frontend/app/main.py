import os
import requests
import streamlit as st
import dotenv

dotenv.load_dotenv()
CHATBOT_URL = os.getenv("CHATBOT_URL")

st.set_page_config(page_title="Random Fortune Telling Bot")
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about NCERT textbook data.
        The agent uses  retrieval-augment generation (RAG).
        """
    )
    st.title("NCERT chatbot")

if "messages" not in st.session_state.keys():
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
            st.markdown(message["output"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"query": prompt}

    with st.spinner("Searching for an answer..."):
        response = requests.post(CHATBOT_URL, json = data)

        if response.status_code == 200:
            output_text = response.json()["answer"]
        else:
            print(response)
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""

    st.chat_message("assistant").markdown(output_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
        }
    )