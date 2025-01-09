from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv() # google API key is in the .env file, loaded in as an environment variable
#setting the model to use

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature = 0.5)


examples = [
    {"text": "Hi, I live in Wisconsin, show me the cheapest plans available", "output": "AREA"},
    {"text": "Does Spectrum work with ACP", "output": ""},
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
    if the user asks any question involving their area, location, zip code, etc. simply output the word AREA.
    Else, for general questions (eg: Does earthlink accept ACP) output nothing.
    If the question is completely irrelevant and has nothing to do with internet, tv, mobile and communications services and providers, please output "TRIPE"
    """

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), MessagesPlaceholder(variable_name="history"), ("user", "{text}")]
)

parser = StrOutputParser()

query_chain = prompt_template | llm | parser

store = {}
user_session = {"configurable": {"session_id": "abc123"}}
user_input = "Please help me find the best internet deal in my area!"

def get_session_history(session_id: str) -> BaseChatMessageHistory: #a function that returns a BaseChatMessageHistory object, 
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        return store[session_id]
        
    else:
        return store[session_id]

query_chain_with_history = RunnableWithMessageHistory(
    query_chain,
    get_session_history,
    input_messages_key="text",
    history_messages_key="history",
)

result = query_chain_with_history.invoke({"text":user_input}, config = user_session)
print(result)



















# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# load_dotenv() # google API key is in the .env file, loaded in as an environment variable
# #setting the model to use

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature = 0.5)


# examples = [
#     {"text": "Hi, I live in Wisconsin, show me the cheapest plans available", "output": "AREA"},
#     {"text": "Does Spectrum work with ACP", "output": ""},
# ]
    
# example_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("human", "{text}"),
#         ("ai", "{output}"),
#     ]
# )

# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=example_prompt,
#     examples=examples,
# )

# system_template = """You are an intent classifier for a cable TV and Internet Service Provider chatbot. Given a user query and the messaging history as context, 
#     if the user asks any question involving their area, location, zip code, etc. simply output the word AREA.
#     Else, for general questions (eg: Does earthlink accept ACP) output nothing.
#     If the question is completely irrelevant and has nothing to do with internet, tv, mobile and communications services and providers, please output "TRIPE"
#     """

# # prompt_template = ChatPromptTemplate.from_messages(
# #     [("system", system_template), few_shot_prompt, MessagesPlaceholder(variable_name="history"), ("user", "{text}")]
# # )

# prompt_template = ChatPromptTemplate.from_messages(
#     [("system", system_template), ("user", "{text}")]
# )

# parser = StrOutputParser()

# query_chain = prompt_template | llm | parser

# print(query_chain.invoke({"text":"What's the internet like in Wisconsin?"}))


# # # def get_query_classified(user_input, bot_memory, user_session):

# #     # store = bot_memory
# #     store = {}
# #     config = {"configurable": {"session_id": "abc2"}}
    
# #     def get_session_history(session_id: str) -> BaseChatMessageHistory: #a function that returns a BaseChatMessageHistory object, 
# #         if session_id not in store:
# #             store[session_id] = ChatMessageHistory()
# #             return store[session_id]
            
# #         else:
# #             return store[session_id]

# #     query_chain_with_history = RunnableWithMessageHistory(
# #         query_chain,
# #         get_session_history,
# #         input_messages_key="text",
# #         history_messages_key="history",
# #     )

# #     result = query_chain_with_history.invoke({"text":user_input}, config = user_session)
# #     return result