from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone
from langchain.schema import Document
import json
import hashlib
from configuration import OPEN_AI_KEY, pc, INDEX_NAME

embedding_model = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)

# Load FAQ data
with open("faqs.json", "r", encoding="utf8") as f:
    data = json.load(f)

docs = []
for category in data["faq_categories"]:
    for sub in category["subcategories"]:
        for entry in sub["entries"]:
            doc = Document(
                page_content=f"Q: {entry['title']}\nA: {entry['answer']}",
                metadata={"category": category["category"], "subcategory": sub["title"]}
            )
            docs.append(doc)

index = pc.Index(INDEX_NAME)

vectors = []
for doc in docs:
    doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
    embedding = embedding_model.embed_query(doc.page_content)
    vectors.append((doc_id, embedding, {"text": doc.page_content}))

index.upsert(vectors)
print(f"Upserted {len(vectors)} documents to Pinecone index: {INDEX_NAME}")
