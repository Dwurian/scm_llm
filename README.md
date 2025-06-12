# SCM-LLM: Creating a Language Model Specific to Supply Chain Management

This is the repository for the LLM project by Dr. Yao Zhao, Dr. Minseok Kim, and Wayne Wu. 

## Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) augments the text generation of language models by retrieving the most relevant piece of information from a database as context. This is done without changing the underlying model, i.e., fine-tuning. There are three main advantage to RAG:
1. Cost: Fine-tuning is a computationally heavy task. It would be very time- and money-consuming to incorporate every piece of new information via fine-tuning. In contrast, RAG allows new information to be easily updated into the database while preserving the original model. 
2. Privacy: RAG allow users to store and retrieve their own private knowledge base (e.g., internal documents, manuals, or reports) without sending the full content to external models. Therefore, users can keep sensitive data local and reduce their exposure. 
3. Traceability: An RAP with proper pipelining can produce identifiers of the retrieved context. As a result, the user can audit the exact retrieved information that was fed into the language model to produce an answer.

### Naive RAG

### Graph RAG

### Agentic RAG

## Note: OpenAI API Key
The files [naive.py](naive.py), [embed_graph.py](embed_graph.py), and [graph_retrieve.py](graph_retrieve.py) uses an OpenAI API Key. Here, the key is stored in a text file named OPENAI_API_KEY.txt, which is absent in this repository for privacy purpose. 
