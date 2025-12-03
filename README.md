# Advanced RAG Concepts: A Layman's Guide

Welcome! This repository contains a notebook (`Advanced_RAG_concepts.ipynb`) that demonstrates how to build a smart "Question Answering" system using **Advanced RAG**.

## What is RAG? (The "Open Book Exam" Analogy)

Imagine you are taking a difficult exam.
- **Traditional AI (like ChatGPT)** is like a student taking the exam from memory. They might hallucinate or get facts wrong if they haven't studied that specific topic recently.
- **RAG (Retrieval-Augmented Generation)** is like an **Open Book Exam**. When you ask a question, the AI first looks up the answer in a textbook (your data), finds the relevant page, and *then* writes the answer. This makes it much more accurate.

## What's "Advanced" about this?

Basic RAG just looks for keywords. This project uses **Advanced RAG** techniques to be much smarter, like a student who doesn't just look at the index but understands the context.

Here are the techniques we use, explained simply:

### 1. Query Transformation ("Rephrasing the Question")
Sometimes, the way you ask a question isn't the best way to find the answer.
- **Concept**: The AI rewrites your question to be clearer or breaks it down into sub-questions.
- **Analogy**: If you ask "How do I fix the thing?", a smart librarian might rephrase it to "Troubleshooting guide for Model X printer".

### 2. Query Routing ("Choosing the Right Tool")
Not all questions need the same search method.
- **Concept**: The AI decides whether to search for *exact keywords* (like a name) or *general concepts* (like "happy movies").
- **Analogy**: If you ask for "Apple stock price", it goes to a finance database. If you ask "Why is the sky blue?", it goes to an encyclopedia.

### 3. Fusion Retrieval ("Combining Search Methods")
We use two ways to search and combine the results.
- **Vector Search**: Looks for *meaning* (e.g., "dog" matches "puppy").
- **Keyword Search**: Looks for *exact words* (e.g., "dog" matches "dog").
- **Analogy**: It's like asking both a poet (who understands metaphors) and a lawyer (who looks for exact wording) to find documents, and then combining their findings.

### 4. Reranking ("The Double Check")
Search engines sometimes bring back too many results, some of which aren't actually good.
- **Concept**: A second, smarter AI reviews the top results and ranks them by how useful they actually are.
- **Analogy**: A research assistant gathers 20 books, but the Professor (the Reranker) reads the summaries and picks the top 3 that *actually* answer the question.

### 5. Context Compression ("Summarizing the Notes")
AI models can only read so much text at once.
- **Concept**: Instead of feeding the AI entire documents, we summarize the relevant parts first.
- **Analogy**: Instead of making you read 5 whole textbooks, I give you a one-page summary of the important facts.

## How to Run This

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Notebook**:
    Open `Advanced_RAG_concepts.ipynb` in Jupyter or VS Code.
3.  **Select Kernel**:
    Make sure to select the `Python (RAG Agent)` kernel if you set it up using the instructions.

## Requirements
- Python 3.10+
- See `requirements.txt` for the full list of libraries.

