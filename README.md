# RAG Retrieval & Reranking Experiment

An experimental Retrieval-Augmented Generation (RAG) project built to understand and demonstrate the difference between **vector retrieval** and **reranking**.

The project implements two RAG pipelines using the same knowledge base, embeddings, and LLM:

* **Baseline RAG:** FAISS → Top-K → Gemini
* **Reranked RAG:** FAISS → Top-N → Cross-Encoder Reranker → Top-K → Gemini

The primary purpose is to understand:

* How vector retrieval works
* Why vector retrieval alone can produce imperfect rankings
* What recall and precision mean in RAG retrieval
* How a Cross-Encoder reranker works
* Why reranking is performed after vector search
* Why we retrieve more candidates before reranking
* How reranking changes document ordering
* Whether better retrieval ordering produces better LLM answers

---

# 1. What Is RAG?

Retrieval-Augmented Generation (RAG) combines information retrieval with a Large Language Model (LLM).

Instead of asking the LLM to answer only from its pretrained knowledge, relevant information is first retrieved from an external knowledge base.

The general pipeline is:

```text
User Query
    ↓
Retrieve Relevant Documents
    ↓
Build Context
    ↓
Send Context + Query to LLM
    ↓
Generate Answer
```

This allows an application to answer questions using its own documents or knowledge base.

---

# 2. Basic RAG Pipeline

The baseline implementation in this project uses vector search.

The pipeline is:

```text
                    USER QUERY
                        │
                        ▼
                 Query Embedding
                        │
                        ▼
                     FAISS
                        │
                        ▼
                  Top-K Chunks
                        │
                        ▼
                  Context Builder
                        │
                        ▼
                     Gemini
                        │
                        ▼
                      Answer
```

The query is converted into an embedding and searched against the embeddings of the document chunks stored in FAISS.

The highest-scoring chunks are then passed to Gemini as context.

---

# 3. Why Do We Need Reranking?

Vector search is extremely useful because it is fast and scalable.

However, vector similarity is not necessarily the same thing as **actual query-document relevance**.

An embedding model represents the semantic meaning of text.

FAISS then compares:

```text
Query Embedding
       ↕
Document Embedding
```

and returns the documents with the highest similarity.

This can work very well, but sometimes a document that is highly relevant to the exact question receives a lower vector similarity score than another document that is only broadly related.

This is where reranking helps.

A reranker performs a second-stage relevance evaluation.

Instead of simply comparing two independent embeddings, a Cross-Encoder evaluates:

```text
Query + Document
       ↓
Cross-Encoder
       ↓
Relevance Score
```

The reranker can therefore reorder the candidates based on their relevance to the exact query.

---

# 4. Two-Stage Retrieval

The reranked pipeline uses two retrieval stages.

```text
                     USER QUERY
                         │
                         ▼
                  Query Embedding
                         │
                         ▼
                      FAISS
                         │
                         ▼
                  Top 20 Candidates
                         │
                         ▼
                Cross-Encoder Reranker
                         │
                         ▼
                   Reranked Results
                         │
                         ▼
                       Top 5
                         │
                         ▼
                      Gemini
                         │
                         ▼
                       Answer
```

The two stages have different responsibilities.

### First Stage — Candidate Retrieval

FAISS searches the complete vector index and retrieves a larger candidate set.

In this experiment:

```text
FAISS → Top 20
```

The objective is to retrieve a broad set of potentially relevant documents.

### Second Stage — Reranking

The Cross-Encoder evaluates those 20 candidates individually against the query.

It then reorders them based on relevance.

Finally:

```text
Top 5
```

are passed to Gemini.

---

# 5. Recall and Precision

Recall and precision are important concepts for understanding why this architecture works.

## Recall

Recall asks:

> How many of the relevant documents did we successfully retrieve?

The formula is:

```text
Recall =
Relevant Documents Retrieved
----------------------------
Total Relevant Documents
```

For example, suppose the knowledge base contains 3 documents relevant to a query.

If FAISS retrieves all 3:

```text
Recall = 3 / 3 = 100%
```

If FAISS retrieves only 2:

```text
Recall = 2 / 3 = 66.7%
```

A reranker cannot recover a document that the first-stage retriever never retrieved.

Therefore, the first-stage retriever should generally prioritize **high recall**.

---

## Precision

Precision asks:

> Of the documents we retrieved, how many are actually relevant?

The formula is:

```text
Precision =
Relevant Documents Retrieved
----------------------------
Total Documents Retrieved
```

For example:

```text
Retrieved documents = 10
Relevant documents  = 3
```

Then:

```text
Precision = 3 / 10 = 30%
```

The retriever found the relevant documents, but also returned many irrelevant ones.

---

# 6. Recall vs Precision in This Project

A useful mental model is:

```text
FAISS
  ↓
"Don't miss potentially relevant information."
```

followed by:

```text
Reranker
  ↓
"Among these candidates, which are actually the most relevant?"
```

Therefore:

```text
First-stage retrieval
→ Fast + broad + high recall

Second-stage reranking
→ More expensive + precise relevance ordering
```

It is common to simplify this as:

```text
Retriever → Recall
Reranker  → Precision
```

However, technically both stages affect retrieval metrics.

The more accurate interpretation is:

> The first stage is optimized for efficiently finding a sufficiently broad candidate set, while the second stage improves the ordering and relevance of those candidates.

---

# 7. Why FAISS Retrieves 20 Instead of 5

One of the most important design decisions in this experiment is:

```text
FAISS → Top 20
```

instead of:

```text
FAISS → Top 5
```

Consider this situation:

```text
FAISS Results

#1 Irrelevant
#2 Irrelevant
#3 Irrelevant
#4 Irrelevant
#5 Irrelevant
...
#18 Highly Relevant
```

If we retrieve only Top-5:

```text
FAISS → Top 5
```

the relevant document is permanently lost.

The reranker never gets a chance to evaluate it.

However:

```text
FAISS → Top 20
```

includes the relevant document in the candidate pool.

The reranker can then recognize its relevance and promote it.

Therefore:

```text
FAISS Top 20
      ↓
Reranker
      ↓
Top 5
```

gives the reranker a larger pool from which to select the final context.

---

# 8. The Reranker Used

This project currently uses:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

from Sentence Transformers.

The model is a Cross-Encoder trained for relevance ranking.

Unlike a normal embedding model, it does not independently encode the query and document and then compare their vectors.

Instead, it receives both pieces of text together:

```text
[Query, Document]
```

and produces a relevance score.

Conceptually:

```text
Query
   +
Document
   ↓
Cross-Encoder
   ↓
Relevance Score
```

Each candidate document receives its own score.

The documents are then sorted according to those scores.

---

# 9. How the Reranker Score Works

Suppose we have:

```text
Query:
Can I get a refund if I cancel my annual subscription?
```

Candidate documents might receive:

```text
Document A → 6.6256
Document B → 6.2910
Document C → 5.6927
Document D → 5.4721
Document E → 3.9848
```

The reranker sorts them:

```text
6.6256
6.2910
5.6927
5.4721
3.9848
```

The higher score indicates that the model considers that query-document pair more relevant relative to the other candidates.

## Important

The reranker score should **not** be treated as a probability.

For example:

```text
6.6256
```

does not mean:

```text
662.56% relevance
```

Likewise, it should not be directly compared with the FAISS score.

FAISS might produce:

```text
0.7522
```

while the reranker might produce:

```text
6.6256
```

These values come from completely different models and scoring systems.

Therefore:

```text
FAISS Score
```

and:

```text
Reranker Score
```

should be treated independently.

For this experiment, we intentionally keep the **raw reranker scores** rather than normalizing them.

---

# 10. Example: Reranking in Action

For the query:

```text
Can I get a refund if I cancel my annual subscription?
```

FAISS initially returned:

```text
1. cancellation.md / chunk 5
   FAISS Score: 0.7657

2. cancellation.md / chunk 6
   FAISS Score: 0.7522

3. subscriptions.md / chunk 5
   FAISS Score: 0.7317

4. refunds.md / chunk 5
   FAISS Score: 0.7218

5. refunds.md / chunk 2
   FAISS Score: 0.7010
```

The important refund-policy chunk was:

```text
refunds.md / chunk 2
```

with:

```text
FAISS Rank = 5
```

After reranking:

```text
1. cancellation.md / chunk 6
   Reranker Score: 6.6256

2. refunds.md / chunk 2
   Reranker Score: 6.2910

3. subscriptions.md / chunk 5
   Reranker Score: 5.6927

4. refunds.md / chunk 5
   Reranker Score: 5.4721

5. cancellation.md / chunk 3
   Reranker Score: 3.9848
```

The refund-policy chunk moved:

```text
FAISS Rank:     #5
Reranker Rank:  #2
```

This is the key behavior we wanted to observe.

The reranker recognized that the refund-policy chunk was highly relevant to the exact question and promoted it.

---

# 11. Why the Answer Did Not Change Much in This Example

An important observation from the experiment is that the final Gemini answer did not change dramatically.

This is expected.

The baseline already retrieved the important document:

```text
refunds.md / chunk 2
```

at FAISS rank #5.

Since the baseline uses Top-5:

```text
FAISS
  ↓
Top 5
  ↓
refunds.md / chunk 2
  ↓
Gemini
```

Gemini already had access to the information required to answer the question.

Therefore, reranking improved the **ordering**, but did not necessarily improve the final answer.

This distinction is important.

A reranker does not guarantee a better answer for every query.

---

# 12. The Ideal Reranking Scenario

The most useful situation is when the relevant document is retrieved by FAISS but ranked outside the final context limit.

For example:

```text
FAISS Top 20

#1 Irrelevant
#2 Irrelevant
#3 Irrelevant
#4 Irrelevant
#5 Irrelevant
...
#18 Relevant
...
```

Baseline:

```text
FAISS
 ↓
Top 5
 ↓
Relevant document excluded
 ↓
Gemini
 ↓
Potentially incomplete answer
```

Reranked pipeline:

```text
FAISS
 ↓
Top 20
 ↓
Reranker
 ↓
Relevant document promoted
 ↓
Top 5
 ↓
Gemini
 ↓
Potentially better answer
```

This is the experiment that best demonstrates the value of reranking.

---

# 13. Baseline vs Reranked Architecture

## Baseline

Implemented in:

```text
rag.py
```

Pipeline:

```text
Markdown Documents
        ↓
     Chunking
        ↓
    Embeddings
        ↓
      FAISS
        ↓
      Top 5
        ↓
     Context
        ↓
      Gemini
        ↓
      Answer
```

---

## Reranked

Implemented in:

```text
rag_reranked.py
```

Pipeline:

```text
Markdown Documents
        ↓
     Chunking
        ↓
    Embeddings
        ↓
      FAISS
        ↓
     Top 20
        ↓
   Cross-Encoder
        ↓
     Reranking
        ↓
      Top 5
        ↓
     Context
        ↓
      Gemini
        ↓
      Answer
```

---

# 14. Project Structure

```text
RAG_experiment/
│
├── data/
│   ├── cancellation.md
│   ├── enterprise.md
│   ├── payments.md
│   ├── refunds.md
│   ├── subscriptions.md
│   └── support.md
│
├── ingest.py
├── retrieve.py
├── reranker.py
├── rag.py
├── rag_reranked.py
├── test_reranker.py
├── config.py
│
├── .env
├── .gitignore
└── README.md
```

---

# 15. File Responsibilities

## `ingest.py`

Responsible for preparing the knowledge base.

Typical workflow:

```text
Markdown files
      ↓
Load documents
      ↓
Chunk documents
      ↓
Generate embeddings
      ↓
Build FAISS index
      ↓
Store metadata
```

---

## `retrieve.py`

Responsible for vector retrieval.

It handles:

* Loading the embedding model
* Loading the FAISS index
* Loading document metadata
* Embedding the query
* Searching FAISS
* Returning ranked candidates

A result generally contains:

```text
source
chunk_id
content
score
```

---

## `reranker.py`

Responsible for loading the Cross-Encoder and reranking retrieved candidates.

Conceptually:

```text
Query
 +
Candidate Documents
      ↓
Cross-Encoder
      ↓
Reranker Scores
      ↓
Sorted Candidates
```

---

## `rag.py`

Contains the baseline RAG pipeline.

```text
FAISS → Top 5 → Gemini
```

This file acts as the control implementation.

It should remain unchanged when comparing against the reranked pipeline.

---

## `rag_reranked.py`

Contains the complete reranked RAG pipeline.

```text
FAISS → Top 20 → Reranker → Top 5 → Gemini
```

---

## `test_reranker.py`

Used to inspect the retrieval and reranking stages without involving Gemini.

It is useful for understanding how rankings change.

Example output:

```text
FAISS RESULTS

#1 ...
#2 ...
#3 ...
```

followed by:

```text
RERANKED RESULTS

#1 ...
#2 ...
#3 ...
```

---

# 16. Running the Project

Activate the virtual environment:

```bash
venv\Scripts\activate
```

If the knowledge base or embeddings need to be regenerated:

```bash
python ingest.py
```

Run the baseline RAG:

```bash
python rag.py
```

Run the reranker experiment:

```bash
python test_reranker.py
```

Run the complete reranked RAG pipeline:

```bash
python rag_reranked.py
```

---

# 17. Environment Variables

The Gemini API key is stored in `.env`.

Example:

```env
GEMINI_API_KEY=your_api_key_here
```

Never commit the `.env` file to Git.

Recommended `.gitignore` entries:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

Generated indexes, local caches, and model artifacts should also be ignored when they can be reproduced from the source documents and code.

---

# 18. Recommended Testing Strategy

When comparing the two systems, use the **same queries**.

For example:

```text
Can I get a refund if I cancel my annual subscription?
```

```text
How long do I have to request a refund for an annual subscription?
```

```text
Does cancelling my subscription automatically give me a refund?
```

```text
What happens to my access when I cancel an annual subscription?
```

```text
What determines refund terms for Enterprise customers?
```

```text
I upgraded from a monthly plan to an annual plan. How is my refund amount determined?
```

The same query should be run through:

```text
rag.py
```

and:

```text
rag_reranked.py
```

This allows a direct comparison.

---

# 19. What Should Be Compared?

Do not compare only the final Gemini answers.

Compare the entire retrieval process.

## 1. FAISS Ranking

Which documents did FAISS retrieve?

```text
FAISS Rank
FAISS Score
Source
Chunk
```

## 2. Reranker Ranking

How did the Cross-Encoder reorder those documents?

```text
Reranker Rank
Reranker Score
Source
Chunk
```

## 3. Context Selection

Which documents actually reached Gemini?

This is particularly important.

## 4. Final Answer

Compare:

* Correctness
* Completeness
* Relevance
* Grounding
* Missing information
* Hallucinations

## 5. Latency

Reranking introduces additional computation.

Compare:

```text
Baseline latency
```

against:

```text
Reranked latency
```

---

# 20. Important Experimental Principle

When evaluating reranking, keep everything else constant.

Do not simultaneously change:

* Chunk size
* Chunk overlap
* Embedding model
* Gemini model
* Gemini prompt
* Number of final documents
* Knowledge base

if the goal is to measure the effect of reranking.

The ideal experiment is:

```text
                   SAME QUERY
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
         BASELINE             RERANKED
             │                   │
         FAISS Top 5         FAISS Top 20
             │                   │
             │                Reranker
             │                   │
             │                Top 5
             │                   │
             └─────────┬─────────┘
                       │
                     Gemini
                       │
                  Compare Results
```

The primary variable is the reranking stage.

---

# 21. Important Limitation

A reranker cannot fix poor first-stage retrieval.

If the relevant document is not present in:

```text
FAISS Top 20
```

then the reranker cannot select it.

This gives us an important rule:

```text
First-stage recall limits second-stage performance.
```

If recall is too low, increasing reranker quality will not solve the problem.

Possible solutions include:

* Increasing FAISS Top-K
* Improving the embedding model
* Improving chunking
* Using hybrid retrieval
* Adding keyword/BM25 retrieval
* Improving metadata filtering

---

# 22. Another Limitation: Reranking Costs More

Vector search is designed to be extremely fast over large collections.

A Cross-Encoder is more computationally expensive because it evaluates each query-document pair.

For example:

```text
FAISS:

Query
 ↓
Vector search
 ↓
20 candidates
```

is relatively inexpensive.

The reranker then performs:

```text
Query + Document 1
Query + Document 2
Query + Document 3
...
Query + Document 20
```

Therefore:

```text
More candidates
    ↓
More reranker computations
    ↓
Higher latency / compute cost
```

This creates a practical tradeoff when choosing:

```text
FAISS candidate count
```

and:

```text
Final context size
```

---

# 23. Why Not Just Send Top 20 to Gemini?

An obvious question is:

> If FAISS found 20 potentially relevant documents, why not just send all 20 to Gemini?

There are several reasons.

More context can mean:

* More irrelevant information
* More redundancy
* More tokens
* Higher latency
* Higher cost
* Greater possibility of distracting the LLM
* More difficult context prioritization

The reranker gives us a way to select a smaller, higher-quality context:

```text
FAISS Top 20
     ↓
Reranker
     ↓
Best 5
     ↓
Gemini
```

The goal is not to retrieve as much information as possible.

The goal is to retrieve **the most useful information for the question**.

---

# 24. Current Experiment Results

For the query:

```text
Can I get a refund if I cancel my annual subscription?
```

the baseline retrieved the relevant refund-policy chunk at:

```text
FAISS Rank #5
```

The reranker promoted it to:

```text
Reranker Rank #2
```

Therefore, the experiment successfully demonstrates that the Cross-Encoder can change the ranking of retrieved documents.

However, because the relevant chunk was already inside the baseline Top-5, the final Gemini answer did not change significantly.

This is an important result rather than a failure.

It demonstrates:

```text
Reranking improved ordering
but
Baseline already had sufficient context
```

The next useful experiment is to test queries where the relevant chunk initially appears below the baseline Top-K but inside the larger reranker candidate pool.

---

# 25. Recommended Future Experiments

## Retrieval Experiments

* Change FAISS candidate count:

  * Top 5
  * Top 10
  * Top 20
  * Top 50
  * Top 100

* Change final context size:

  * Top 3
  * Top 5
  * Top 10

---

## Chunking Experiments

Test different:

* Chunk sizes
* Chunk overlaps
* Chunking strategies
* Semantic chunking
* Markdown-aware chunking
* PDF-specific chunking

---

## Model Experiments

Compare:

* Different embedding models
* Different Cross-Encoder models
* Different reranking models
* Different Gemini models

---

## Retrieval Strategy Experiments

Eventually compare:

```text
Vector Search
```

against:

```text
Hybrid Search
```

For example:

```text
BM25 + Vector Search
       ↓
Candidate Pool
       ↓
Reranker
       ↓
Top-K
```

---

# 26. Evaluation Metrics

Once the project moves beyond manual experimentation, introduce quantitative evaluation.

Useful metrics include:

### Recall@K

Measures whether relevant documents appear within the first K results.

```text
Recall@K
```

is especially useful for evaluating the first-stage retriever.

---

### Precision@K

Measures how many of the first K results are relevant.

```text
Precision@K
```

is useful for evaluating the quality of the selected context.

---

### Mean Reciprocal Rank (MRR)

Measures how highly the first relevant result appears.

If the first relevant result is:

```text
Rank 1 → 1.0
Rank 2 → 0.5
Rank 3 → 0.333
Rank 4 → 0.25
```

then averaging this across queries gives MRR.

MRR is particularly useful for studying whether reranking successfully moves relevant documents toward the top.

---

# 27. Final Mental Model

The entire project can be remembered with this simple architecture:

```text
                         QUERY
                           │
                           ▼
                    Query Embedding
                           │
                           ▼
                         FAISS
                           │
                    Find Candidates
                           │
                         Top 20
                           │
                           ▼
                      RERANKER
                           │
                Evaluate Relevance
                           │
                         Top 5
                           │
                           ▼
                        GEMINI
                           │
                           ▼
                         ANSWER
```

The responsibilities are:

```text
FAISS
→ Fast candidate discovery

Reranker
→ Better relevance ordering

Gemini
→ Generate the final grounded answer
```

---

# 28. The Core Question of This Experiment

The purpose of this project is not simply:

> "Can a reranker produce a different score?"

The deeper question is:

> **Does adding a second-stage relevance model allow the RAG system to provide better context to the LLM than vector similarity alone?**

The experiment should therefore ultimately answer:

```text
Does:

FAISS → Top 5 → Gemini

perform worse than:

FAISS → Top 20 → Reranker → Top 5 → Gemini
```

when the relevant information is difficult for vector search to rank correctly?

That is the central experiment this repository is designed to explore.
