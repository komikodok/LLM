from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.prompts.chat import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

import json 
import os
from pymongo import MongoClient

with open("config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
mongodb_config = config["mongodb"]

def load_llm() -> ChatOllama:
    return ChatOllama(model=model_config["repo_id"])

def load_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model_config["repo_id"])

def mongo_client():
    client = MongoClient(mongodb_config['mongodb_connection'])
    db_name = mongodb_config['db_name']
    collection_name = mongodb_config['collection_name']
    collection = client[db_name][collection_name]
    index_name = mongodb_config['index_name']
    return collection, index_name

def load_retriever(embeddings):
    collection, index_name = mongo_client()
    vectorstore = MongoDBAtlasVectorSearch(embedding=embeddings, collection=collection, index_name=index_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

def prompt_template(key: str):
    key_string = ["history-prompt", "rag_prompt"]  
    if key == str.lower("history-prompt"):    
        history_prompt = """
            Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", history_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    elif key == str.lower("rag-prompt"):
        rag_prompt = """
            You are name is Ruby. \
            You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\

            {context}
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", rag_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    else:
        raise ValueError(f"Invalid value: {key}, Allowed values are: {key_string}")

def rag_chain():
    history_prompt = prompt_template(key="history-prompt")
    llm = load_llm()
    parser = StrOutputParser()
    qa_history_chain = history_prompt | llm | parser
    retriever = load_retriever(embeddings=load_embeddings())
    retriever_chain = RunnablePassthrough.assign(
        context=qa_history_chain
                | retriever
                | (lambda docs: "\n\n".join([doc.page_content for doc in docs]))
    )

    rag_prompt = prompt_template(key="rag-prompt")
    rag_chain = (
        retriever_chain
        | rag_prompt
        | llm
    )
    return rag_chain

def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        connection_string=mongodb_config['mongodb_connection'],
        session_id=session_id,
        database_name=mongodb_config['db_name'],
        collection_name=mongodb_config['collection_history']
    )

def runnable_message_history(chain):
    return RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history'
    )

class ChatChain():


    def __init__(self):
        self.rag_chain = rag_chain()
        self.with_message_history = runnable_message_history(chain=self.rag_chain)

    def invoke(self, user_input: str, id="1"):
        self.id = {"configurable": {"session_id": id}}
        return self.with_message_history.invoke({'input': user_input}, config=self.id)


if __name__ == "__main__":

    chain = ChatChain()
    question = input("You: ")
    result = chain.invoke(user_input=question)
    print(f"Assistant: {result}")