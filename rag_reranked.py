from google import genai

from config import GEMINI_API_KEY
from retrieve import load_resources, retrieve
from reranker import load_reranker, rerank


# -----------------------------
# Configuration
# -----------------------------

GENERATION_MODEL = "gemini-3.6-flash"

FAISS_TOP_K = 20
FINAL_TOP_K = 5


# -----------------------------
# Gemini Client
# -----------------------------

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# -----------------------------
# Build Context
# -----------------------------

def build_context(results):

    context_parts = []

    for rank, result in enumerate(results, start=1):

        context_parts.append(
            f"""
--- Context {rank} ---
Source: {result['source']}
Chunk: {result['chunk_id']}

{result['content']}
"""
        )

    return "\n".join(context_parts)


# -----------------------------
# Generate Answer
# -----------------------------

def generate_answer(query, results):

    context = build_context(results)

    prompt = f"""
You are a helpful assistant answering questions using a
company knowledge base.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer
the question, clearly say that the available information is
insufficient.

Do not invent or assume information that is not present in
the context.

User question:
{query}

Knowledge base context:
{context}
"""

    chat = client.chats.create(
        model=GENERATION_MODEL
    )

    response = chat.send_message(
        message=prompt
    )

    return response.text


# -----------------------------
# Main
# -----------------------------

def main():

    # -------------------------
    # Load retrieval resources
    # -------------------------

    embedding_model, index, metadata = load_resources()

    # -------------------------
    # Load reranker
    # -------------------------

    reranker = load_reranker()

    print("\n" + "=" * 70)
    print("RERANKED RAG SYSTEM")
    print("=" * 70)

    print("\nType 'exit' to quit.")

    while True:

        query = input("\nEnter your question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        # -------------------------
        # Stage 1: FAISS Retrieval
        # -------------------------

        results = retrieve(
            query=query,
            model=embedding_model,
            index=index,
            metadata=metadata,
            top_k=FAISS_TOP_K
        )

        print("\n" + "=" * 70)
        print("FAISS CANDIDATES")
        print("=" * 70)

        for rank, result in enumerate(results, start=1):

            print(
                f"\n[{rank}] "
                f"FAISS Score: {result['score']:.4f}"
            )

            print(f"Source: {result['source']}")
            print(f"Chunk:  {result['chunk_id']}")
            print(f"Content: {result['content']}")

        # -------------------------
        # Stage 2: Reranking
        # -------------------------

        reranked_results = rerank(
            query=query,
            results=results,
            model=reranker
        )

        # -------------------------
        # Select final context
        # -------------------------

        final_results = reranked_results[:FINAL_TOP_K]

        print("\n" + "=" * 70)
        print("RERANKED TOP 5")
        print("=" * 70)

        for rank, result in enumerate(
            final_results,
            start=1
        ):

            print(
                f"\n[{rank}] "
                f"Reranker Score: "
                f"{result['reranker_score']:.4f}"
            )

            print(
                f"Original FAISS Score: "
                f"{result['score']:.4f}"
            )

            print(f"Source: {result['source']}")
            print(f"Chunk:  {result['chunk_id']}")
            print(f"Content: {result['content']}")

        # -------------------------
        # Stage 3: Gemini
        # -------------------------

        print("\n" + "=" * 70)
        print("GEMINI ANSWER")
        print("=" * 70)

        answer = generate_answer(
            query=query,
            results=final_results
        )

        print(answer)


if __name__ == "__main__":
    main()