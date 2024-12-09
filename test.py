import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import time
st.set_page_config(page_title = "CableBot", page_icon=":material/rocket:")

st.header("BestCable Assistant 🤖")
st.write("Write any question you may have about our bundles, deals and offers, and your helpful assistant will try it's best to answer your queries!")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature = 0.7)


system_prompt = (
    """You are a helpful, polite AI assistant who will help users find the ideal internet, TV, or bundle deal for them. Use the provided document's data to help aid you in answering their questions. 
    
    Keep the answer concise and to the point while still answering the user's question properly and including all relevant info (such as bundle provider name, cost, speed, which phone number to call, etc.)
    Use the history provided to get context and guide your answers.

    If the user asks about how you can order, simple guide them to click the call now button above, or give them a call to action to call 877-395-5851 for ordering or more information.
    If the question is completely irrelevant to TV, internet, Phone, Mobile plans, bundles, and deals, please give an apologetic message informing them that you can only help them with the mentioned topics.
    If you truly cannot answer the user question, direct them to call our agents at 877-395-5851 for more information.""" 
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
if original_question := st.chat_input("Ask Away!: "):
    with st.chat_message("assistant"):
        st.write("Let me find that for you...")
        results = question_answer_chain.run(original_question)
        st.write_stream(results);


