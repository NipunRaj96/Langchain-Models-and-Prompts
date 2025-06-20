from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=0.1,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

result = model.invoke("so who i am")
print(result.content)
