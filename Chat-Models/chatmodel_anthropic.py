from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="Claude-3-5-sonnet")
result = model.invoke("what is the capital of india?")

print(result.content)