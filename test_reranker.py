from retrieve import load_resources, retrieve
from reranker import load_reranker, rerank


FAISS_TOP_K = 20
FINAL_TOP_K = 5


def main():

    # -------------------------
    # Load retrieval resources
    # -------------------------

    embedding_model, index, metadata = load_resources()

    # -------------------------
    # Load reranker
    # -------------------------

    reranker = load_reranker()

    print("\nReranker ready.")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nEnter your question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        # -------------------------
        # Stage 1: FAISS
        # -------------------------

        results = retrieve(
            query=query,
            model=embedding_model,
            index=index,
            metadata=metadata,
            top_k=FAISS_TOP_K
        )

        print("\n" + "=" * 70)
        print("FAISS RESULTS")
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
        # Display reranked results
        # -------------------------

        print("\n" + "=" * 70)
        print("RERANKED RESULTS")
        print("=" * 70)

        for rank, result in enumerate(
            reranked_results[:FINAL_TOP_K],
            start=1
        ):

            print(
                f"\n[{rank}] "
                f"Reranker Score: "
                f"{result['reranker_score']:.4f}"
            )

            print(f"Original FAISS Score: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Chunk:  {result['chunk_id']}")
            print(f"Content: {result['content']}")


if __name__ == "__main__":
    main()