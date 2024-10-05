import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
st.set_page_config(page_title = "CableBot", page_icon=":material/rocket:")
from StickyAssistant import sticky_container

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
from faq_helper import get_faq_result
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
    
#re-print entire chat history each time file re-runs.
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#returns words one by one with small time span
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

    with st.spinner('Your assistant is thinking...'):
        #classify the query to choose which bot to send it to
        query_class = get_query_classified(original_question, st.session_state.bot_memory, st.session_state.user_session) # Have the bot recognize user intent from their query

        #area query
        if "AREA" in query_class:
            results, st.session_state.bot_memory = get_sql_result(original_question, st.session_state.bot_memory, st.session_state.user_session)
            if("FAQ" in results): # If the answer cannot be sufficiently answered by the SQL bot, attempt to answer it with the PDF bot
                results, st.session_state.bot_memory = get_faq_result(original_question, st.session_state.bot_memory, st.session_state.user_session)

        elif "TRIPE" in query_class: #Totally irrelevant question
            results = "I'm very sorry, but I can only answer questions about TV, Internet, and phone services, along with information about their providers. Thank you for understanding.";

        #faq query
        else:
            results, st.session_state.bot_memory = get_faq_result(original_question, st.session_state.bot_memory, st.session_state.user_session)

    with st.chat_message("assistant"):
        st.write_stream(response_generator(results))
    st.session_state.history.append({"role": "assistant", "content": results})

    st.rerun() #To ensure the output from stream is formatted correctly
