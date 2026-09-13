from pathlib import Path
import pickle

import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

INDEX_PATH = Path("faiss.index")
METADATA_PATH = Path("metadata.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5


# -----------------------------
# Load Resources
# -----------------------------

def load_resources():
    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading FAISS index...")

    index = faiss.read_index(str(INDEX_PATH))

    print("Loading metadata...")

    with open(METADATA_PATH, "rb") as file:
        metadata = pickle.load(file)

    return model, index, metadata


# -----------------------------
# Retrieve Documents
# -----------------------------

def retrieve(query, model, index, metadata, top_k=TOP_K):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):

        if index_position == -1:
            continue

        chunk = metadata[index_position]

        results.append({
            "score": float(score),
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"]
        })

    return results


# -----------------------------
# Display Results
# -----------------------------

def print_results(query, results):

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)

    print(query)

    print("\n" + "=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\n[{rank}]")
        print(f"Source: {result['source']}")
        print(f"Chunk:  {result['chunk_id']}")
        print(f"Score:  {result['score']:.4f}")

        print("\nContent:")
        print(result["content"])

        print("-" * 70)


# -----------------------------
# Main
# -----------------------------

def main():

    model, index, metadata = load_resources()

    print("\nRAG retrieval system ready.")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nEnter your question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = retrieve(
            query=query,
            model=model,
            index=index,
            metadata=metadata,
            top_k=TOP_K
        )

        print_results(query, results)


if __name__ == "__main__":
    main()