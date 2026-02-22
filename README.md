# RAG Chatbot — Read Text Files & Answer Questions

An intelligent chatbot that reads your own text files and answers
questions about them using Retrieval Augmented Generation (RAG).
It combines FAISS vector search with the Groq LLaMA3 language model
to find relevant information and generate accurate answers.


## How It Works
```
Your Question
      |
      v
Convert to Vector
      |
      v
Search FAISS -> Top 3 Relevant Chunks
      |
      v
Send (Question + Chunks) to Groq LLM
      |
      v
Get Answer
```


## Models and Tools
| Tool | Purpose | Why |
|---|---|---|
| Sentence Transformers | Convert text to vectors | Captures meaning not just keywords |
| FAISS | Store and search vectors | Fast similarity search by Meta |
| Groq LLaMA3 | Generate answers | Free and extremely fast LLM |
| Python 3.10 | Main language | Clean and simple code |
