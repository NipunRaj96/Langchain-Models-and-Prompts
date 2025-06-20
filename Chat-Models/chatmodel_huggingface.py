from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1", #llm you want to use from huugging face site
    task = "text-generation",
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("Who is the president of india?")


print(result.content)