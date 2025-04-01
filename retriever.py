from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import openai
import json
from pinecone import Pinecone
from configuration import OPEN_AI_KEY, pc, INDEX_NAME

embedding_model = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)
index = pc.Index(INDEX_NAME)

openai.api_key = OPEN_AI_KEY

conversation_history = [
        {"role": "system", "content": "You are a helpful FAQ assistant named ROTS (Reliable Online Troubleshooting System )."},
        {"role": "system", "content": "You use the language that are used by the user, with Bahasa Indonesia as the default language"},
        {"role": "system", "content": "You answer questions based on the context provided and can handle follow-up questions conversationally"},
        {"role": "system", "content": "Only answer topics regarding the services provided by KofKom and topic regarding network and telecommunicatons, don't answer every other topics"},
        {"role": "system", "content": "When the user start troubleshooting something, you can response by asking more question regarding the problem, if you already sure of the problem answer in details how to solve the problems"},
        {"role": "system", "content": "If there are procedure regarding kofkom services that are not yet defined, don't start making things up, just say you don't know and be apologetic, don't add some general procedure from other apps"},
    ]

def handle_query(query, conversation_history):
    query_embedding = embedding_model.embed_query(query)

    results = index.query(vector=query_embedding, top_k=1, include_metadata=True)


    relevant_entries = [
        {
            "title": match["metadata"]["text"].split('\n')[0].replace("Q: ", "").strip(),
            "answer": match["metadata"]["text"].split('\n')[1].replace("A: ", "").strip()
        }
        for match in results["matches"]
    ]

    context = "\n".join([f"Q: {entry['title']}\nA: {entry['answer']}" for entry in relevant_entries])
    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "system", "content": f"Relevant context:\n{context}"})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )

    ai_response = response['choices'][0]['message']['content']
    conversation_history.append({"role": "assistant", "content": ai_response})

    return ai_response, conversation_history

print("Start chatting with the FAQ bot! Type 'exit' to end the conversation.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response, conversation_history = handle_query(user_input, conversation_history)
    print(f"Bot: {response}\n")
