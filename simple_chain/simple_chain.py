from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import json
import os


with open("config.json", "r") as f:
    config = json.load(f)
    
model_config = config["model"]

llm = ChatOllama(model=model_config["repo_id"])
template = '''
            You're expert AI assistant for answer-question task. \
            If the user question asks in Indonesian, the answer must be answered in Indonesian. \
        '''
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{question}")
    ]
)

chain = prompt | llm
question = input("You: ")
result = chain.invoke({'question': question})

print(result.content)


