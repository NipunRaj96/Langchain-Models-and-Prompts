from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,AIMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model=ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage(content="You are an helpful ai assistant")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content="user_input"))
    if user_input=="exit":
        break;
    result=model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)