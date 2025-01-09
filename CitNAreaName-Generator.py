import ast #Abstract Syntax tree
import re # REgular expressions
from langchain_community.utilities import SQLDatabase

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
import sqlite3


# ______________________________________________________SQL to DB_______________________________________________________________________________
# Create a new SQLite database .db file (empty)
# conn = sqlite3.connect('cable_TV_database.db')
# cursor = conn.cursor()

# # Example of executing SQL statements individually
# sql_files = ['cable_data.sql']


# for file_name in sql_files: # .sql data copied to .db
#     with open(file_name, 'r') as file:
#         sql_script = file.read().split(';')  # Split statements by semicolon
#         for statement in sql_script:
#             statement = statement.strip()
#             if statement:  # Execute non-empty statements
#                 try:
#                     cursor.execute(statement)
#                 except sqlite3.OperationalError as e:
#                     print(f"Error executing statement: {statement}\n{e}")

# # Commit the changes to the database and close the connection
# conn.commit()
# conn.close()
# ______________________________________________________SQL to DB END_______________________________________________________________________________

database = SQLDatabase.from_uri("sqlite:///cable_TV_database.db") 

# runs a database query and extracts proper nouns (in this case cities and areas) from the results, storing them in a list. 
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

cities = query_as_list(database, "SELECT city FROM plans")
areas = query_as_list(database, "SELECT area FROM plans")
print(cities[:5])

relevant_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

cities_and_areas = FAISS.from_texts(areas + cities, relevant_embeddings) # creates the vector store of all city and area names in database
retriever = cities_and_areas.as_retriever(search_kwargs={"k": 5})
description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
valid proper nouns. Use the noun most similar to the search."""
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)

cities_and_areas.save_local("cities_and_areas_names") # stores the vector store

#For reference, below is the loader: 
# cities_and_areas = FAISS.load_local("cities_and_areas_names", embeddings = HuggingFaceInstructEmbeddings(), allow_dangerous_deserialization = True)