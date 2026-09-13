from pathlib import Path
import pickle

import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

DOCS_DIR = Path("company_docs")
INDEX_PATH = Path("faiss.index")
METADATA_PATH = Path("metadata.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# -----------------------------
# Load Documents
# -----------------------------

def load_documents():
    documents = []

    for file_path in DOCS_DIR.glob("*.md"):
        content = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "content": content
        })

    return documents


# -----------------------------
# Chunk Documents
# -----------------------------

def chunk_document(document):
    paragraphs = document["content"].split("\n\n")

    chunks = []

    for index, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        chunks.append({
            "source": document["source"],
            "chunk_id": index,
            "content": paragraph
        })

    return chunks


def create_chunks(documents):
    all_chunks = []

    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)

    return all_chunks


# -----------------------------
# Create Embeddings
# -----------------------------

def create_embeddings(chunks, model):
    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


# -----------------------------
# Create FAISS Index
# -----------------------------

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# -----------------------------
# Save Everything
# -----------------------------

def save_index(index, chunks):
    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "wb") as file:
        pickle.dump(chunks, file)


# -----------------------------
# Main
# -----------------------------

def main():
    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    print("Creating chunks...")

    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings...")

    embeddings = create_embeddings(chunks, model)

    print(f"Embedding shape: {embeddings.shape}")

    print("Creating FAISS index...")

    index = create_faiss_index(embeddings)

    print(f"FAISS index contains {index.ntotal} vectors.")

    print("Saving index and metadata...")

    save_index(index, chunks)

    print("\nIngestion completed successfully!")
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()