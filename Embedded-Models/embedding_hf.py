from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")

#text = "Delhi is the capital of India"
#vector= embeddings.embed_query(text)

#for large no. of sentence or data
documents = [
    "Delhi is the capital of india",
    "France is the capital of paris",
    "i live in new delhi"
]
vector = embeddings.embed_documents(documents)
print(str(vector))