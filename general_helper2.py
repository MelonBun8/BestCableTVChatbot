import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import time

#setting the model to use
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature = 0.7)

instructor_embeddings = HuggingFaceInstructEmbeddings()

def custom_trimmer(my_list):
    n = len(my_list)
    if n == 0:
        return my_list
    elif n > 0 and n < 7:
        return my_list[-n:]
    else:
        return my_list[-8:]

def get_general_result(user_input, bot_memory, user_session):

    store = bot_memory
    
    vectorDB = FAISS.load_local("FAQ_PDF_VectSto", instructor_embeddings, allow_dangerous_deserialization=True)
    retriever = vectorDB.as_retriever()

    ( # now before we 
 


    )

    def get_session_history(session_id: str) -> BaseChatMessageHistory: #a function that returns a BaseChatMessageHistory object, 
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
                return store[session_id]
                
            else:
                # Retrieve current messages
                current_messages = store[session_id].messages
                # Keep only the last 4 messages
                latest_messages = custom_trimmer(current_messages)
                
                # Clear existing messages
                store[session_id].clear()
                
                # Add the latest messages back to the history
                for message in latest_messages:
                    store[session_id].add_message(message)
            
                return store[session_id]

    system_prompt = (
        """You are a helpful, polite AI assistant who will help users find the ideal internet, TV, or bundle deal for them. Use the provided document's data to help aid you in answering their questions. 
        
        Keep the answer concise and to the point while still answering the user's question properly and including all relevant info (such as bundle provider name, cost, speed, which phone number to call, etc.)
        
        If the user asks about how you can order, simple guide them to click the call now button above, or give them a call to action to call 877-395-5851 for ordering or more information.""" 
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"), 
            ("human", "{input}")
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    rag_chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
        )

    full_output = rag_chain_with_history.invoke({"input": user_input}, config=user_session)
    results = full_output['answer']
    return results, store
