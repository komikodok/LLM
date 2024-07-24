from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.runnables import RunnablePassthrough

import json
from pymongo import MongoClient

with open("config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
mongodb_config = config["mongodb"]

embeddings = OllamaEmbeddings(model=model_config["repo_id"])

def mongo_client():
    client = MongoClient(mongodb_config['mongodb_connection'])
    db_name = mongodb_config['db_name']
    collection_name = mongodb_config['collection_name']
    collection = client[db_name][collection_name]
    index_name = mongodb_config['index_name']
    return collection, index_name

collection, index_name = mongo_client()
vectorstore = MongoDBAtlasVectorSearch(embedding=embeddings, collection=collection, index_name=index_name)
retriever = vectorstore.as_retriever()

class Schema(BaseModel):

    context: str = Field(description="Context the answer")
    content: str = Field(description="Answer the question")


parser = JsonOutputParser(pydantic_object=Schema)

template = """
            You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\
            
            Format instructions: {format_instructions}. \
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "Question: {question}")
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

format_prompt = prompt.partial(format_instructions=parser.get_format_instructions(), question=RunnablePassthrough())
format_prompt.format(context=retriever | format_docs)

llm = ChatOllama(model=model_config["repo_id"])


rag_chain = (
    format_prompt 
    | llm 
    | parser
)

result = rag_chain.invoke({'question': 'what is langchain expression language?'})
print()
print(result)
print(type(result))
print()
