from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

import json
from pymongo import MongoClient

with open("config.json", "r") as f:
    config = json.load(f)

def load_knowledge(path: str):
    knowledge = path.split('.')
    if knowledge[-1] == 'txt':
        loader = TextLoader('.'.join(knowledge))
        txt_docs = loader.load()
        return txt_docs
    elif knowledge[-1] == 'pdf':
        loader = PyPDFLoader('.'.join(knowledge))
        pdf_docs = loader.load_and_split()
        return pdf_docs
    else:
        loader = WebBaseLoader(path)
        web_docs = loader.load()
        return web_docs

def load_document(document):
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n'], chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(document)
    return chunks

def docs2mongodb(knowledge):
    client = MongoClient(config['mongodb']['mongodb_connection'], serverSelectionTimeoutMS=5000)
    db_name = config['mongodb']['db_name']
    collection_name = config['mongodb']['collection_name']
    collection = client[db_name][collection_name]
    embeddings = OllamaEmbeddings(model='custom-llama3')
    docs2mongodb = MongoDBAtlasVectorSearch.from_documents(documents=load_document(load_knowledge(knowledge)), embedding=embeddings, collection=collection)
    return docs2mongodb

docs2mongodb("<copy_relative_path_document>")