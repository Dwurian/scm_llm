# SCM-LLM: Creating a Language Model Specific to Supply Chain Management

In recent years, large language models (LLMs) has garnered much attention in the mainstream due to several highly visible products such as ChatGPT and Deepseek. Beyond general purpose LLMs, there are several domain-specific language models in fields such as finance (e.g., FinBERT), law (e.g., LegalBERT), and medicine (e.g., BioBERT). By finetuning with domain specific data, such models can achieve better performance in their respective fields than general purpose language models ([Araci 2019](https://doi.org/10.48550/arXiv.1908.10063), [Chalkidis et al 2020](https://doi.org/10.48550/arXiv.2010.02559), [Lee et al 2020](https://doi.org/10.1093/bioinformatics/btz682)). Many are also smaller and easier to deploy due to a reliance of simpler basic structures like BERT. 

In this project, we aim to create a domain-specific language model for supply chain management. We will use Retrieval-Augmented Generation to enable multi-task and multi-hop reasoning, 

## Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) augments the text generation of language models by retrieving the most relevant piece of information from a database as context. This is done without changing the underlying model, i.e., fine-tuning. There are three main advantage to RAG:

1. **Update Cost**: Fine-tuning is a computationally heavy task. It would be very time- and money-consuming to incorporate every piece of new information via fine-tuning. In contrast, RAG allows new information to be easily updated into the database while preserving the original model. 
2. **Privacy**: RAG allow users to store and retrieve their own private knowledge base (e.g., internal documents, manuals, or reports) without sending the full content to external models. Therefore, users can keep sensitive data local and reduce their exposure. 
3. **Traceability**: An RAG with proper pipelining can produce identifiers of the retrieved context. As a result, the user can audit the exact retrieved information that was fed into the language model to produce an answer.

In our project, we study three forms of retrievers: Naive RAG, Graph RAG, and Agentic RAG. We discuss their concepts below. As of June 20, 2025, [Naive RAG](naive.py) and [a version of Graph RAG](graph_retrieve.py) have been prototyped. 

### Naive Retriever

The Naive Retriever works by finding the most similar text to the prompt from the database, and feeding them alongside the prompt to the language model as context. Here, the vector embeddings (the dense vectors representing the original text) of the text chunks in the database are compared with the vector embedding of the prompt in a process called similarity search. In similarity search, relevancy is determined by purely mathematical operations, such as minimum Euclidean distance, on the vector embeddings. A pipeline can be seen in the figure below: 

[figure here]

A Naive Retriever is the simplest way of enjoying the benefits of RAG. However, because of the simple structure, it is incapable of multihop reasoning or providing extra functionality to the language model. Also, since its underlying mechanisms are mathematical, its ability to produce the most relevant information cannot be improved without changing the text embedding model. 

As of June 20, 2025, we have finised a [Naive Retriever prototype](naive.py).

### Graph Retriever

The Graph Retriever uses graphs' ability of representing relationships to improve the retrieval performance. In our specific usage, graphs can be used to represent: 

1. Supply chain relationships, where nodes are companies, and edges represent supplier-buyer relationships.
2. Text chunk relationships, where nodes are text chunks, and edges represent the structure of the original document and citations.

As of June 20, 2025, we focus on the second case, where Graph Retriever replaces the Naive Retriever for text retrieval. This enables multi-hop reasoning by enriching our first level context with secondary context that expand on their ideas. An example can be seen below: 

[figure here]

Here, the Mom article is a journal paper by Zhang et al ([2022](https://doi.org/10.1287/msom.2022.1117)) about 3D printing's impact on spare part logistics. In a subsection of introduction, Hanaphy's ([2021](https://3dprintingindustry.com/news/cadchains-new-boris-plug-in-protects-users-designs-during-product-rd-197876/)) blogpost is referenced as an example of IP licensing technology necessary for using the 3D printing in spare part production. By tracing this citation relationship in the graph, the relevant text in the blogpost is retrieved and new information, such as the company's name, the product's name, and the underlying blockchain technology, is retrieved, even though the original text in the blogpost does not directly resemble the prompt. 

As of June 20, 2025, we have finised a [Graph Retriever prototype](graph_retrieve.py), which produced the example above. However, we need a way to extract references from articles and automatically embed citation relationships into the graph. Our current [Graph Embedder prototype](embed_graph.py) can embed article structures, but still requires manual citation embedding. 

### Agentic Retriever

Agentic RAG uses an AI agent as the retriever, which is capable of breaking down prompts and calling tools. This is a truly powerful retriever that can combine multiple sources (document search, calculation tools, translators, etc.) to provide a detailed context. To achieve this, the agent should be able to generate a tract of actions and make corresponding tool calls, making Chain-of-Thought (CoT) or CoT-like models suitable for this task. Below is an ideal use case of Agentic RAG: 

[figure here]

Notably, document search should still be a subfunction of the Agentic Retriever, meaning that the Agentic RAG should include Graph Retriever and/or Naive Retriever as a subfunction. 

As of June 20, 2025, Agentic Retriever is still in the ideation phase. Interesting references include: 

1. **ReAct** ([Yao et al 2023](https://doi.org/10.48550/arXiv.2210.03629)) which uses prompt engineering to encourage GPT, a non-CoT model, to generate CoT-like action tracts.
2. **Toolformer** ([Schick et al 2023](https://doi.org/10.48550/arXiv.2302.04761)) which uses a self-supervising structure to fine-tune a language model that includes tool calls as part of the generated text.

## Fine-Tuning

In large language models, fine-tuning is tweaking the model parameters to generate preferable answers when prompted. As LLMs such as GPT generate text by predicting the next word token based on the prompt and previously generated tokens via probability, this involves changing the underlying probability distributions of the model. In our context, doing so requires two steps: 
1. **General Domain Adaptation**, where the model is fed raw domain-specific text so the model is better at predicting the next word based on previous words. In this step, the model learns the domain volabulary, syntax, and knowledge. This is particularly important for jargon-heavy fields such as law and finance, and fields with uncommon words such as medicine. The purpose of this step is for the model to "talk" like a professional in this field. This step is unsupervised.
2. **Task-Specific Fine-Tuning**, where labeled data is used so that the model learns to perform a specific task: classification, Q&A, summarization, etc. The purpose of this step is for the model to "make sense" in its response. This step is supervised.

In fine-tuning, data is crucial. For our purpose, text for general domain adaptation can be found with web-crawled data from academic and new articles. However, data for the second step might be tricky and needs further investigation. 

## Distillation

Distillation is the process of producing a smaller language model, often called a student model, from a model with more parameters, conversely called the teacher model. To do so, the student model is fed the prompt and answer of the teacher, mimicking the teacher's behavior. The ideal distilled model can achieve similar performance to the teacher while being more lightweight and easier to deploy. 

Notably, distillation can happen before or after fine-tuning, with each option having their own advantage. 

- **Distilling a Fine-Tuned Model**
  * **Full Fine-Tuned Teacher Model:** By fine-tuning before distillation, we can have a full domain-specific LLM, which would not exist in the other option. 
  * **Better Performance Potential of Students:** Since we can have a complete LLM suitable for our needs, so we know our small models are trained by a language models that reliably cater to our specific needs. 
- **Fine-tuning a distilled model**
  * **Cheaper fine tuning:** As mentioned previously, fine-tuning is computationally demanding, and a distilled model with fewer parameters can be more cost effective both in terms of time and money. 
  * **Available pre-distilled model:** There are many openly available distilled models such as the [Llama series](https://www.llama.com) by Meta and the [Qwen series](https://qwenlm.github.io) by Alibaba.
 
## Summary of Progress

1. **Retrieval Augmented Generation**
  - **Naive RAG** (Prototype done)
  - **Graph RAG** (Prototype mostly done)
    * **Graph Retriever** (Prototype done)
    * **Graph Embedder** (Prototpe partially done)
  - **Agentic RAG** (Ideation)
2. **Fine-Tuning** (Ideation)
3. **Distillation** (Ideation)

## Note: OpenAI API Key
The files [naive.py](naive.py), [embed_graph.py](embed_graph.py), and [graph_retrieve.py](graph_retrieve.py) uses an OpenAI API Key stored in a text file named OPENAI_API_KEY.txt, which is absent in this repository for privacy purpose. 
