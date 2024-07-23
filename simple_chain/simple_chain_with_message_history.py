from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import json


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
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ]
)

chain = prompt | llm
chat_history = []

for i in range(3):
    question = input("You: ")

    result = chain.invoke({"question": question, "chat_history": chat_history})
    answer = result.content

    h_message = HumanMessage(content=question)
    a_message = AIMessage(content=answer)
    
    chat_history.append(h_message)
    chat_history.append(a_message)


    print(answer)


