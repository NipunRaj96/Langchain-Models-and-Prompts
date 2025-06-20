from langchain_core.messages import SystemMessage,AIMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model=ChatHuggingFace(llm=llm)
messages=[
    SystemMessage(content="You are a very helful and knowledgeable teacher of maths , you answer everything with 2-3 lines of explanation"),
    HumanMessage(content="Tell me the relation between lcm and hcf")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)