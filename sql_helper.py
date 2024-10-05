import streamlit as st
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Example of executing SQL statements individually
# sql_files = ['cable_data.sql']
db = SQLDatabase.from_uri("sqlite:///cable_TV_database.db")

# load_dotenv() # google API key is in the .env file, loaded in as an environment variable
#setting the model to use
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature = 0.0)

examples_SQL = [
    {"input": "Does Spectrum have any deals in Washington? ", "result": """SELECT provider, phone1, internet_type, speed, phone1, price, availibility, rating FROM plans WHERE state = "Washington" AND provider = "Spectrum" LIMIT 5"""},
    {"input": "Cheapest deals in Frankfort? ", "result": """SELECT provider, phone1, internet_type, speed, phone1, min(price), availibility, rating FROM plans WHERE city='Frankfort' GROUP BY provider LIMIT 5"""},
    {"input": "I'm looking for the fastest deal in Wyoming", "result": """SELECT provider, phone1, internet_type, max(speed), phone1, price, availibility, rating FROM plans WHERE state='Wyoming' GROUP BY provider LIMIT 5"""}
]

example_SQL_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{result}"),
    ]
)

few_shot_SQL_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_SQL_prompt,
    examples=examples_SQL,
)

def custom_trimmer(my_list):
    n = len(my_list)
    if n == 0:
        return my_list
    elif n > 0 and n < 7:
        return my_list[-n:]
    else:
        return my_list[-8:]

cities_and_areas = FAISS.load_local("cities_and_areas_names", embeddings = HuggingFaceInstructEmbeddings(), allow_dangerous_deserialization = True)

def get_sql_result(user_input,bot_memory,user_session):
    # user_session = {"configurable": {"session_id": "abc123"}}
    store = bot_memory

    retriever = cities_and_areas.as_retriever(search_kwargs={"k": 5})
    description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
    valid proper nouns. Use the noun most similar to the search."""
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description=description,
    )
    
    tools = [retriever_tool]
    tooly_llm = llm.bind_tools(tools)

    query_fixer_template = """You are a query editor for a cable TV and Internet Service Provider chatbot. Given a user query, if it involves a U.S. state, zip code, country or larger, return the word 'BOB'.
    However, if the query mentions a particular area, city or location, please use the 'search_proper_nouns' tool to find and match the closest
    possible word"""

    query_fixer_prompt_template = ChatPromptTemplate.from_messages(
        [("system", query_fixer_template), ("user", "{text}")]
    )

    closest_words_chain = query_fixer_prompt_template | tooly_llm
    tool_response = closest_words_chain.invoke(user_input)

    if(tool_response.content == ""):
        
        only_call = tool_response.tool_calls[0]
        tool_argument = only_call['args']['query'] #extract the argument / parameter the tool call from the LLM returned.
        closest_words = retriever_tool.invoke(tool_argument)

        edited_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Users can make spelling mistakes. Given a query with list of words, replace the misspelt location/area/city name in the query with the closest name from the list of words provided:{closest_list}. 
                    Return JUST the new corrected query""",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        corrected_query_chain = edited_prompt | llm

        corrected_query_response = corrected_query_chain.invoke(
            {"messages": [HumanMessage(content= user_input)], "closest_list": closest_words}
        )

        user_input = corrected_query_response

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

    my_prompt_template = """You are an SQLite expert. Given an input question, create a syntactically correct SQL statement compatible with sqlite.
    Unless specified, query for at most 5 results using the LIMIT clause.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Sometimes you will need to use the history (previous chat messages) to know what the user wants. Keep it in mind as context when generating the query: {history}
    You have access to a SQLite database about internet plans. Each row contains information about specific deals/plans/bundles. 
    When giving information about plans, the query MUST includes the columns: [ provider, phone1, internet_type, speed, phone1, price, availibility, rating ] 
    the state and city should have a capital first letter, while area must be lower case.

    If the query does not mention provider, you MUST use the GROUP BY provider clause to get one unique plan per provider

    Please return only the statement itself For example: 'SELECT DISTINCT zip FROM internet_type LIMIT 5;'

    Only use the following tables:
    {table_info}

    Question: {input}

    Number of Results: {top_k}
    """

    # my_prompt = PromptTemplate(
    #     template = my_prompt_template, input_variables = ["input","table_info","top_k","history"]
    #     ) 

    with_examples_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", my_prompt_template),
        few_shot_SQL_prompt,
        ("human", "{input}"),
    ]
    )

    with_examples_prompt.input_variables = ["input","table_info","top_k","history"]

    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db, with_examples_prompt)

    # IN_chain = write_query | execute_query

    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question. Try to stay polite but concise.
        If the SQL result is corrupted, empty, or does not answer the user's query at all, simply output the word "FAQ"

        If the user is asking about plans and bundles, and the SQL query has returned multiple rows of data, format your answer like the below example:

        EXAMPLE:

        Question: "Cheapest deals in Frankfort?"
        SQL Query: SELECT provider, phone1, internet_type, speed, phone1, min(price), availibility, rating FROM plans WHERE city='Frankfort' GROUP BY provider LIMIT 5
        SQL Result: [('AT&T', '8772-0935-74', 'DSL , Fiber', '5000 Mbps', '$55/mo', '80.3% availability', 4), ('EarthLink', '8772-0924-67', 'DSL, Fiber, Wireless, Satellite', '1000 Mbps', '$69.95/mo', '80.3% availability', 3), ('HughesNet', '8772-0924-59', 'Satellite', '25 Mbps', '$49.99/mo', '80.3% availability', 3), ('Spectrum', '877-410-3834', 'Hybrid fiber-coaxial', '1000 Mbps', '$49.99/mo', '80.3% availability', 5), ('Viasat', '877-412-0759', 'Satellite', '100 Mbps', '$39.99/mo', '100% availability', 4)]

        Ideal answer: Okay, here are the cheapest plans offered by providers in Frankfort:

        **AT&T**
        > 5000 Mbps for $55/mo | Internet Type: DSL , Fiber | Availability: 80.3% | Rating: 4 (of 5 stars) | CALL 8772-0935-74 now for more info!

        **EarthLink**
        > 1000 Mbps for $69.95/mo | Internet Type: DSL, Fiber, Wireless, Satellite | Availability: 80.3% | Rating: 3 (of 5 stars) | CALL 8772-0924-67 now for more info!

        ... (and so on for remaining plans)

        ----END OF EXAMPLE----

        If the user enters their state, city or area, always prompt them to enter their zip code in the last line. If they have entered their zip code already, ask if you
        can help them with anything else.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    OUT_chain = (
        RunnablePassthrough.assign(query=write_query).assign(result=itemgetter("query") | execute_query)
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    sql_chain_with_history = RunnableWithMessageHistory(
        OUT_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    result = sql_chain_with_history.invoke({"question":user_input,"history":store}, config = user_session)
    return result, store
