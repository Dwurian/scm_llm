import os
import openai
import faiss
import numpy as np
import pickle
import tiktoken

# Load API Key
OPENAI_API_KEY = "OPENAI_API_KEY.txt"
with open(OPENAI_API_KEY, "r", encoding="utf-8") as file:
    openai.api_key = file.read()

# Define Paths
CHUNK_DIR = "chunk/abstract"
EMBED_STORE = "embed_store/index.faiss"
TEXTS_STORE = "embed_store/texts.pkl"

# Function to get embeddings
def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])

# Create or load FAISS index
d = 1536  # Dimension for ada-002 embeddings
if os.path.exists(EMBED_STORE):
    index = faiss.read_index(EMBED_STORE)
else:
    index = faiss.IndexFlatL2(d)

# Read and embed chunks
if os.path.exists(TEXTS_STORE):
    with open(TEXTS_STORE, "rb") as f:
        texts = pickle.load(f)
else:
    texts = []

def embed_texts():
    print(f"Current chunk directory: {CHUNK_DIR}")
    for filename in os.listdir(CHUNK_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(CHUNK_DIR, filename), "r", encoding="utf-8") as file:
                text = file.read()
                try:
                    embedding = get_embedding(text)
                    index.add(np.array([embedding]))
                    texts.append(text)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
                if (len(texts) % 100 == 0):
                    print(f"Processed {len(texts)} texts")
                    print(f"Vector store object length: {index.ntotal}")
    print("Finished embedding texts")
    print(f"Processed {len(texts)} texts")
    print(f"Vector store object length: {index.ntotal}")

def estimate_tokens(text, encoding):
    return len(encoding.encode(text))

def batch_embed(text_list, batch_size=100, model="text-embedding-3-small", max_tokens=8000):
    encoding = tiktoken.encoding_for_model(model)
    valid_texts = []
    skipped_indices = []

    # Filter out overlong texts
    for idx, text in enumerate(text_list):
        if estimate_tokens(text, encoding) <= max_tokens:
            valid_texts.append(text)
        else:
            skipped_indices.append(idx)
            print(f"Skipped index {idx} (too long)")

    # Batch embedding
    embeddings = []
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        response = openai.embeddings.create(
            input=batch,
            model=model
        )
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)

    return valid_texts, np.array(embeddings).astype("float32"), skipped_indices

CHUNK_DIR = "chunk/newchunks2"
embed_texts()

# Save FAISS index
faiss.write_index(index, EMBED_STORE)

with open(TEXTS_STORE, "wb") as f:
    pickle.dump(texts, f)
print(f"Total chunks embedded: {len(texts)}")


def search_index(query, k=5):
    if index.ntotal == 0:
        raise ValueError("FAISS index is empty. Generate embeddings first.")

    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)

    print(f"FAISS returned indices: {indices[0]}")  # Debugging
    print(f"Total text chunks: {len(texts)}")  # Debugging

    matched_chunks = [texts[i] for i in indices[0] if 0 <= i < len(texts)]

    if not matched_chunks:
        return "No relevant results found."

    return "\n\n".join(matched_chunks)

def query_chatgpt(query):
    # Search for relevant context
    context = search_index(query, k = 10)

    # Call ChatGPT with context
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    return response['choices'][0]['message']['content'].strip()


# Example usage
user_question = "According to the provided context, what are the current trends of inventory management research?"
answer = query_chatgpt(user_question)
print("Answer:", answer)
