"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os

os.environ['OPENAI_API_KEY']='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'
os.environ['SERPAPI_API_KEY']='07b053c02f0440e32630980fb3efe1437216c746d94ee9f465a2fa8d95683bd9'

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI


def load_agent():
    """Logic for loading the chain you want to use should go here."""
    # First, let's load the language model we're going to use to control the agent.
    chat = ChatOpenAI(temperature=0)

    # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)


    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent

agent = load_agent

# From here down is all the StreamLit UI.
st.set_page_config(page_title="有能な新人君GPT", page_icon=":robot:")
st.header("有能な新人君GPT")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "あなたは有能な新人君です。いまから調査を依頼しますので、自分なりに調べて結論をまとめてください。", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = agent.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
