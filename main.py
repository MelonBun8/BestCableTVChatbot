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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature = 0.0)

def db_lazy_loader(classified_db):
    if( (f'{classified_db}') not in st.session_state):
        temp = FAISS.load_local((f'vector_stores//{classified_db}'), embeddings, allow_dangerous_deserialization=True)
        exec(f'st.session_state.{classified_db} = temp')
        st.write((classified_db + "has been loaded for the first time!"))
        return temp
    else:
        st.write((classified_db + "Has already been loaded before! Using one in memory..."))
        return( exec(f'st.session_state.{classified_db}'))

def database_classifier(user_input):
    database_index = ['att_bundle', 'att_internet', 'cable_internet', 'direct_tv', 'dish_tv', 'dsl_internet', 'earthlink_internet', 'fiber_internet', 'five_g_internet'
                      ,'fixed_wireless_internet', 'frontier_bundle', 'frontier_internet', 'general_bundle', 'general_internet', 'general_tv', 'hughesnet_internet', 
                      'ipbb_internet', 'optimum_bundle', 'optimum_internet', 'optimum_tv', 'satellite_internet', 'spectrum_bundle', 'spectrum_internet',
                      'spectrum_tv', 'verizon_bundle', 'verizon_internet', 'viasat_internet', 'windstream_bundle', 'windstream_internet']
    
    
    examples = [
        {"text": "Hi, please tell me about Optimum's internet deals", "output": "optimum_internet"},
        {"text": "How many channels can I get with dish TV?", "output": "dish_tv"},
    ]
        
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{text}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    system_template = """You are an intent classifier for a cable TV and Internet Service Provider chatbot. Given a user query and the messaging history as context,
        return the most relevant database name from the following database names, return ONLY the database name itself, no quotation marks.
    
        'att_bundle', 'att_internet', 'cable_internet', 'direct_tv', 'dish_tv', 'dsl_internet', 'earthlink_internet', 'fiber_internet', 'five_g_internet'
                      ,'fixed_wireless_internet', 'frontier_bundle', 'frontier_internet', 'general_bundle', 'general_internet', 'general_tv', 'hughesnet_internet', 
                      'ipbb_internet', 'optimum_bundle', 'optimum_internet', 'optimum_tv', 'satellite_internet', 'spectrum_bundle', 'spectrum_internet',
                      'spectrum_tv', 'verizon_bundle', 'verizon_internet', 'viasat_internet', 'windstream_bundle', 'windstream_internet'    
        """
    
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), few_shot_prompt, ("user", "{text}")]
    )
    
    parser = StrOutputParser()
    
    query_chain = prompt_template | llm | parser

    grouping = query_chain.invoke({"text":user_input})
    st.write("Grouped into" + grouping)
    grouping = grouping.strip()
    relevant_data = db_lazy_loader(grouping)
    return relevant_data

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
                results, st.session_state.bot_memory = get_general_result(original_question, st.session_state.bot_memory, st.session_state.user_session)

        elif "TRIPE" in query_class: #Totally irrelevant question
            results = "I'm very sorry, but I can only answer questions about TV, Internet, and phone services, along with information about their providers. Thank you for understanding.";

        #faq query
        else:
            relevant_data = database_classifier(original_question)
            results, st.session_state.bot_memory = get_general_result(original_question, st.session_state.bot_memory, st.session_state.user_session, relevant_data)

    with st.chat_message("assistant"):
        st.write_stream(response_generator(results))
    st.session_state.history.append({"role": "assistant", "content": results})

    st.rerun() #To ensure the output from stream is formatted correctly
