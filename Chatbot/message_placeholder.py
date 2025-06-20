from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import SystemMessage,AIMessage, HumanMessage

#chat template -
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support bot'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history=[]
with open('/Users/hariomkumar/Playground/Langchain-Models/Chatbot/chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({'chat_history':chat_history, 'query':'where is my refund?'})

print(prompt)