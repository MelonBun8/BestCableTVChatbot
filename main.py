import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Set the page title and icon
st.set_page_config(page_title = "CableBot", page_icon=":material/rocket:")
from StickyAssistant import sticky_container

# Set the title and subtitle of the page content
st.header("BestCable Assistant 🤖")
st.write("Write any question you may have about our bundles, deals and offers, and your helpful assistant will try it's best to answer your queries!")
with sticky_container(mode="top", border=True):
    st.write("Wish to talk to a live agent? click the button below!")
    st.link_button("Call now!", "tel:+18773955851",type="primary")  
    
st.session_state.user_session = {"configurable": {"session_id": "abc123"}}

# loading bar for loading up the data 
percent_complete = 0
progress_text = "Your assistant is loading... Please wait."
my_bar = st.progress(0, text=progress_text)
from query_assigner import get_query_classified
my_bar.progress(percent_complete + 33, text=progress_text)
percent_complete+=33
from sql_helper import get_sql_result
my_bar.progress(percent_complete + 33, text=progress_text)
percent_complete+=33
from general_helper import get_general_result
my_bar.progress(percent_complete + 34, text=progress_text)
time.sleep(1)
my_bar.empty()


#Initialize the chatbot's memory
if "bot_memory" not in st.session_state:
    st.session_state.bot_memory = {}

#initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.history.append({"role": "assistant", "content": "Hello there, what can I help you with today? (You can ask me about plans, bundles, and deals!)"})
    

from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature = 0.0)

#trim the chat history to only the last 4 messages (user and bot pairs)
def custom_trimmer(my_list):
    n = len(my_list)
    if n == 0:
        return my_list
    elif n > 0 and n < 7:
        return my_list[-n:]
    else:
        return my_list[-8:]

# Load the database from the local storage on-demand only
def db_lazy_loader(classified_db):
    classified_db = classified_db.strip()
    if( (f'{classified_db}') not in st.session_state):
        temp = FAISS.load_local((f'vector_stores\\{classified_db}'), HuggingFaceInstructEmbeddings(), allow_dangerous_deserialization=True)
        st.session_state[classified_db] = temp
        st.write((classified_db + "has been loaded for the first time!"))
        return temp
    else:
        st.write((classified_db + "Has already been loaded before! Using one in memory..."))
        temp = st.session_state.get(classified_db, None)
        if(temp == None):
            st.write("Error in loading the pre-loaded database")
        return( temp )
#_______________________________________________________________________________________________________________
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# Data model
class RouteVectorDBQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal ["att_bundle", "att_internet", "cable_internet", "direct_tv", "dish_tv", "dsl_internet", "earthlink_internet", "fiber_internet", "five_g_internet"
                      ,"fixed_wireless_internet", "frontier_bundle", "frontier_internet", "general_bundle", "general_internet", "general_tv", "hughesnet_internet", 
                      "ipbb_internet", "optimum_bundle", "optimum_internet", "optimum_tv", "satellite_internet", "spectrum_bundle", "spectrum_internet",
                      "spectrum_tv", "verizon_bundle", "verizon_internet", "viasat_internet", "windstream_bundle", "windstream_internet"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

# LLM with function call
database_classifying_llm = llm.with_structured_output(RouteVectorDBQuery)

# Classifies the general query to the correct database, then calls db_lazy_loader to load the database
def database_classifier(user_input):
    store = st.session_state.bot_memory
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
        

    system_template = """You are an intent classifier (through routing) for a cable TV and Internet Service Provider chatbot. Given a user query and the messaging history as context,
        route it to the most relevant database name. Given below are the database names for reference.

        "att_bundle", "att_internet", "cable_internet", "direct_tv", "dish_tv", "dsl_internet", "earthlink_internet", "fiber_internet", "five_g_internet"
                      ,"fixed_wireless_internet", "frontier_bundle", "frontier_internet", "general_bundle", "general_internet", "general_tv", "hughesnet_internet", 
                      "ipbb_internet", "optimum_bundle", "optimum_internet", "optimum_tv", "satellite_internet", "spectrum_bundle", "spectrum_internet",
                      "spectrum_tv", "verizon_bundle", "verizon_internet", "viasat_internet", "windstream_bundle", "windstream_internet"

        No matter how irrelevant the query may seem, if you are unsure of the intent, route it to one of the general databases 
        ('general_bundle', 'general_internet', or 'general_tv')

        DO NOT EVER return None, or route to NoneType, ALWAYS route to something.

        Use the {history} to get context and better understand the user's intent.

        EXAMPLE:
        User: "Hi, can you please tell me about optimum internet deals?"
        You: routed to optimum_internet
        User: "Do they accept ACP?"
        You: routed to optimum_internet (Used the history to classify the query)
        """
    
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), MessagesPlaceholder(variable_name="history"), ("user", "{input}")]
    )
    
    query_chain = prompt_template | database_classifying_llm 

    query_chain_with_history = RunnableWithMessageHistory(
        query_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        )
    
    database_classifier_answer = query_chain_with_history.invoke({"input": user_input}, config=st.session_state.user_session)
    # grouping = query_chain.invoke({"text":user_input})
    grouping = database_classifier_answer.datasource
    st.write("Grouped into" + grouping)

    relevant_data = db_lazy_loader(grouping)
    return relevant_data

#______________________________________________________________________________________________________________________________
#re-print entire chat history each time file re-runs (to ensure customer can see the entire chat history)
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# returns words one by one with small time span
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

#If user asks a question:
if original_question := st.chat_input("Ask Away!: "):
    
    # Display user input and save it to the state history
    with st.chat_message("user"):
        st.markdown(original_question)
    st.session_state.history.append({"role": "user", "content": original_question})

    # Display a loading spinner while the assistant is thinking
    with st.spinner('Your assistant is thinking...'):
        #classify the query to choose which bot to send it to
        query_class = get_query_classified(original_question, st.session_state.bot_memory, st.session_state.user_session) # Have the bot recognize user intent from their query

        #area query
        if "AREA" in query_class:
            st.write("Classified into AREA!!!\n\n")
            results, st.session_state.bot_memory = get_sql_result(original_question, st.session_state.bot_memory, st.session_state.user_session)
            if("FAQ" in results): # If the answer cannot be sufficiently answered by the SQL bot, attempt to answer it with the PDF bot
                relevant_data = database_classifier(original_question)
                results, st.session_state.bot_memory = get_general_result(original_question, st.session_state.bot_memory, st.session_state.user_session, relevant_data)

        elif "TRIPE" in query_class: #Totally irrelevant question
            results = "I'm very sorry, but I can only answer questions about TV, Internet, and phone services, along with information about their providers. Thank you for understanding.";

        # general query
        else:
            relevant_data = database_classifier(original_question)
            results, st.session_state.bot_memory = get_general_result(original_question, st.session_state.bot_memory, st.session_state.user_session, relevant_data)

    # Display the assistant's response and save it to the state history
    with st.chat_message("assistant"):
        st.write_stream(response_generator(results))
    st.session_state.history.append({"role": "assistant", "content": results})

    st.rerun() #To ensure the output from stream is formatted correctly
