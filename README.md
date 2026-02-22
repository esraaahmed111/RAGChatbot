# RAG Chatbot — Read Text Files & Answer Questions

An intelligent chatbot that reads your own text files and answers
questions about them using Retrieval Augmented Generation (RAG).
It combines FAISS vector search with the Groq LLaMA3 language model
to find relevant information and generate accurate answers.

---

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

---

## Models and Tools

| Tool | Purpose |
|---|---|
| Sentence Transformers | Convert text to vectors |
| FAISS | Store and search vectors |
| Groq LLaMA3 | Generate answers |
| Python | Main language |


## Project Structure
```
rag-chatbot/
│
├── rag_chatbot.py       <- main code
├── requirements.txt     <- libraries
├── README.md            <- you are here
└── documents/           <- put your .txt files here
    ├── ai_basics.txt
    ├── python_basics.txt
    └── nlp_topics.txt


## Chat Commands

| Command | Action |
|---|---|
| Any question | Get answer from your documents |
| history | See all previous Q&A |
| clear | Reset conversation memory |
| quit | Exit the chatbot |
