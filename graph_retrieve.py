import os
import openai
import faiss
import numpy as np
import pickle
import tiktoken
import networkx as nx


# Load API Key
OPENAI_API_KEY = "OPENAI_API_KEY.txt"
with open(OPENAI_API_KEY, "r", encoding="utf-8") as file:
    openai.api_key = file.read()

DATABASE_DIR = "graph_db/demo3"

STORE_DIR = os.path.join(DATABASE_DIR, "embed_store")

ID_STORE = os.path.join(STORE_DIR, "id.pkl")
EMBED_STORE = os.path.join(STORE_DIR, "index.faiss")
TEXTS_STORE = os.path.join(STORE_DIR, "texts.pkl")
GRAPH_STORE = os.path.join(STORE_DIR, "graph.pkl")

PRINT_STEP = True

EMBEDDING_MODEL = "text-embedding-ada-002"
LLM = "gpt-4o"

index = faiss.read_index(EMBED_STORE)
with open(TEXTS_STORE, "rb") as f:
    texts = pickle.load(f)
with open(GRAPH_STORE, "rb") as f:
    G = pickle.load(f)
with open(ID_STORE, "rb") as f:
    num_to_id, id_to_num = pickle.load(f)

# Function to get embeddings
def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return np.array(response['data'][0]['embedding'])

def search_index(query, allowed_indices, k=2):
    if index.ntotal == 0:
        raise ValueError("FAISS index is empty. Generate embeddings first.")

    if len(allowed_indices) == 0:
        raise ValueError("No available chunk left to search.")

    query_embedding = get_embedding(query).reshape(1, -1)

    id_selector = faiss.IDSelectorArray(allowed_indices)
    distances, indices = index.search(query_embedding, k, params=faiss.SearchParametersIVF(sel=id_selector))

    if PRINT_STEP:
        print(f"FAISS returned indices: {indices[0]}")  # Debugging
        print(f"Corresponding chunk IDs are: {[num_to_id[i] for i in indices[0] if 0 <= i < len(texts)]}")
        #print(f"Total text chunks considered: {len(allowed_indices)}")  # Debugging

    matched_IDs = [num_to_id[i] for i in indices[0] if 0 <= i < len(texts)]

    if not matched_IDs:
        return "No relevant results found."

    return indices[0], matched_IDs

def iter_search(query, max_steps, k_list):
    context_IDs = []
    search_range = range(len(texts))

    max_possible_context_size = 0
    max_layer_size = 1
    for x in k_list:
        max_layer_size *= x
        max_possible_context_size += max_layer_size

    if PRINT_STEP:
        print("Searching layer 1, range: all indices")
    returned_indices, returned_IDs = search_index(query, search_range, k = k_list[0])
    context_IDs = context_IDs + returned_IDs
    new_layer_IDs = returned_IDs

    for i in range(max_steps-1):

        if PRINT_STEP:
            print(f"\n\nSearching layer {i+2}, searching the neighbors of: {new_layer_IDs}")
        newer_layer_IDs = []

        for ID in new_layer_IDs:
            # Get neighbors of node ID
            neighbors = set(G.neighbors(ID))

            # Get neighbors of those neighbors
            neighbors_of_neighbors = set()
            for n in neighbors:
                neighbors_of_neighbors.update(G.neighbors(n))

            # Remove the original node and its direct neighbors to get pure 2-hop nodes
            neighbors_of_neighbors.discard(1)
            neighbors_of_neighbors -= neighbors

            search_IDs = list(neighbors_of_neighbors - set(context_IDs))
            search_range = [id_to_num[k] for k in search_IDs]

            if PRINT_STEP:
                print(f"\nSearching neighbors of {ID}, index range: {search_range}, corresponding IDs: {search_IDs}")

            try:
                returned_indices, returned_IDs = search_index(texts[ID], search_range, k = k_list[i+1])
                context_IDs = list(dict.fromkeys(context_IDs + returned_IDs))
                newer_layer_IDs += returned_IDs
            except Exception as e:
                continue

        # Update the nodes of interest for the next search
        else:
            new_layer_IDs = newer_layer_IDs

    print(f"\n\nTotal text chunk in store: {len(texts)}")
    print(f"Maximum possible context size: {max_possible_context_size}")
    print(f"Actual context size: {len(context_IDs)}")
    print(f"Context IDs: {context_IDs}")
    return context_IDs


def query_chatgpt_graph(query, max_steps = 3, k = 2):

    if isinstance(k, int):
        k_list = [k] * max_steps
    elif isinstance(k, (list, tuple, np.ndarray)) and np.ndim(k) == 1:
        if len(k) != max_steps:
            raise ValueError("When specifying k with a one-dimensional array-like object, the length of k should be the same as max_steps")
        if not all(isinstance(x, int) for x in k):
            raise TypeError("All elements in k must be integers")
        k_list = list(k)
    else:
        raise TypeError("k should be an integer or a one-dimensional array-like object")

    # Search for relevant context
    context_IDs = iter_search(query, max_steps, k_list)

    context_chunks = [texts[i] for i in context_IDs if i in texts]

    context = "\n\n".join(context_chunks)

    # Call ChatGPT with context
    response = openai.ChatCompletion.create(
        model=LLM,
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    return response['choices'][0]['message']['content'].strip()

def query_chatgpt_naive(query, k = 5):
    # Search for relevant context
    indices, context_IDs = search_index(query, allowed_indices= range(len(texts)), k = k)

    context_chunks = [texts[i] for i in context_IDs if i in texts]

    context = "\n\n".join(context_chunks)

    # Call ChatGPT with context
    response = openai.ChatCompletion.create(
        model=LLM,
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    return response['choices'][0]['message']['content'].strip()

def query_chatgpt_basic(query):
    response = openai.ChatCompletion.create(
        model=LLM,
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering the following question."},
            {"role": "user", "content": f"Question: {query}"}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    return response['choices'][0]['message']['content'].strip()

user_question = "What is an example of IP licensing technology that enables the transition to 3D printing for spare part production? "

# max_steps denotes how many "hops" the retriever should take
# k can either be 1) a list of numbers dictating how many related chunks to grab in each hop,
# in which case the length of the list must equal max_steps, or 2) an integer number, in which
# case all steps would grab the same number of chunks

# Here max_steps = 2 and k = [3,2], so the retriever would perform 2 hops, first hop grabbing
# 3 most related chunks to the query, and second hop grabbing 2 chunks most related to each
# of the 3 chunks from the first hop
graph_answer = query_chatgpt_graph(user_question, max_steps = 2, k = 2)
print("\n\nAnswer from Graph RAG:", graph_answer)

naive_answer = query_chatgpt_naive(user_question, k = 6)
print("\n\nAnswer from Naive RAG:", naive_answer)

basic_answer = query_chatgpt_basic(user_question)
print("\n\nAnswer from no RAG:", basic_answer)