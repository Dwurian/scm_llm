import os
import openai
import faiss
import numpy as np
import pickle
import tiktoken
import networkx as nx
import matplotlib.pyplot as plt

# Load API Key
OPENAI_API_KEY = "OPENAI_API_KEY.txt"
with open(OPENAI_API_KEY, "r", encoding="utf-8") as file:
    openai.api_key = file.read()

#######################
##### DEMO making #####
#######################


# Define Paths

DATABASE_DIR = "graph_db/demo3"

CHUNK_DIR = os.path.join(DATABASE_DIR, "chunk")
STORE_DIR = os.path.join(DATABASE_DIR, "embed_store")

os.makedirs(STORE_DIR, exist_ok=True)

ID_STORE = os.path.join(STORE_DIR, "id.pkl")
EMBED_STORE = os.path.join(STORE_DIR, "index.faiss")
TEXTS_STORE = os.path.join(STORE_DIR, "texts.pkl")
GRAPH_STORE = os.path.join(STORE_DIR, "graph.pkl")

DEMO = 3

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
    base_index = faiss.IndexFlatL2(d)
    index = faiss.IndexIDMap(base_index)

# Create or load chunk storage
if os.path.exists(TEXTS_STORE):
    with open(TEXTS_STORE, "rb") as f:
        texts = pickle.load(f)
else:
    texts = {}

# Create or load graph
if os.path.exists(GRAPH_STORE):
    with open(GRAPH_STORE, "rb") as f:
        G = pickle.load(f)
else:
    G = nx.Graph()

# Create or load ID storage
if os.path.exists(ID_STORE):
    with open(ID_STORE, "rb") as f:
        num_to_id, id_to_num = pickle.load(f)
else:
    num_to_id = []
    id_to_num = {}

def embed_texts_from_paper(name):
    refID = name
    dir = os.path.join(CHUNK_DIR, refID)
    print(f"Current chunk directory: {dir}, corresponding to the paper {refID}")
    G.add_node(refID, type = "paper")
    for filename in os.listdir(dir):
        if filename.endswith(".txt"):
            with open(os.path.join(dir, filename), "r", encoding="utf-8") as file:
                text = file.read()
                try:
                    embedding = get_embedding(text)
                    index.add_with_ids(np.array([embedding]),len(num_to_id))
                    textID = os.path.splitext(filename)[0]
                    id_to_num[textID] = len(num_to_id)
                    num_to_id.append(textID)
                    texts[textID] = text
                    G.add_node(textID, type = "text")
                    print(f"Adding node {textID} to graph")
                    G.add_edge(textID, refID)
                    print(f"Adding edge between {textID} and {refID}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
                if (len(texts) % 100 == 0):
                    print(f"Processed {len(texts)} texts")
                    print(f"Vector store object length: {index.ntotal}")
    #print("Finished embedding texts")
    #print(f"Processed {len(texts)} texts")
    #print(f"Vector store object length: {index.ntotal}")

for name in os.listdir(CHUNK_DIR):
    full_path = os.path.join(CHUNK_DIR, name)
    if os.path.isdir(full_path):
        embed_texts_from_paper(name = name)

all_paper_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'paper']

print("Finished embedding texts")
print(f"Processed {len(texts)} texts from {len(all_paper_nodes)} papers")
print(f"Vector store object length: {index.ntotal}")

################################
### Manual Citation Tracking ###
################################

# Demo 1
if DEMO == 1:

    G.add_edge("dong2022_liter", "song2020")
    G.add_edge("dong2022_concl_1", "song2020")
    G.add_edge("zhang2022_liter", "song2020")
    G.add_edge("chen2021_liter", "song2020")
    G.add_edge("chen2021_intro", "song2020")
    G.add_edge("song2020_liter", "chen2021")
    G.add_edge("dong2022_liter", "chen2021")
    G.add_edge("zhang2022_liter", "chen2021")
    G.add_edge("zhang2022_liter", "dong2022")

    others = list(set(all_paper_nodes) - set(["sethuraman2023"]))
    for i in others:
        G.add_edge("sethuraman2023_liter", i)
        liter = i + "_liter"
        G.add_edge("sethuraman2023", liter)

if DEMO == 3:
    G.add_edge("mom_1","child1")
    G.add_edge("mom_2", "child2")
    G.add_edge("mom_3", "child3")
    G.add_edge("mom_4", "child4")

def draw_graph():
    color_map = []
    for node in G.nodes:
        node_type = G.nodes[node].get("type", "")
        if node_type == "text":
            color_map.append("skyblue")
        elif node_type == "paper":
            color_map.append("salmon")
        else:
            color_map.append("gray")  # fallback

    nx.draw_networkx(G, with_labels=True, node_color=color_map, pos=nx.spring_layout(G))
    plt.show()

draw_graph()

################
##### Save #####
################

# Save FAISS index
faiss.write_index(index, EMBED_STORE)

# Save text
with open(TEXTS_STORE, "wb") as f:
    pickle.dump(texts, f)
print(f"Total chunks embedded: {len(texts)}")

# Save graph
with open(GRAPH_STORE, "wb") as f:
    pickle.dump(G, f)
print(f"Total number of nodes: {G.number_of_nodes()}, total number of edges: {G.number_of_edges()}")

# Save ID
with open(ID_STORE, "wb") as f:
    pickle.dump((num_to_id, id_to_num), f)
print(num_to_id)