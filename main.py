"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

def load_chain(urls):
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    urls = ['https://qiita.com/yazoo/items/dfbdefb2bbb23352e743']
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return chain

urls = [st.text_input('url')]

if urls:
    qa = load_chain(urls)
else:
    pass

# From here down is all the StreamLit UI.
st.set_page_config(page_title="ChatVEC", page_icon=":robot:")
st.header("ChatVEC")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "what is this about?", key="input")
    return input_text


user_input = get_text()
ask_button = st.button('ask')

def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)

if ask_button:
    chat_history = []
    result = qa({"question": user_input, "chat_history": chat_history})#, get_chat_history=get_chat_history)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])
    # chat_history.append(user_input)
    # chat_history.append(result)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
