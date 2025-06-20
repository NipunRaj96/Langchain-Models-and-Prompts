from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

document = [
"Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
"MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
"Sachin Tendulkar, also known as the 'God of Cricket, holds many batting records.",
"Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
"Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = input("whom do you wanna know about - ")

doc_embeddings = embeddings.embed_documents(document)
query_embeddings = embeddings.embed_query(query)

scores = cosine_similarity([query_embeddings],doc_embeddings)[0]

index , score = sorted(list(enumerate(scores)), key=lambda x:x[-1])[-1]

print(query)
print(document[index])
print(f"Similarity score is: {score}")