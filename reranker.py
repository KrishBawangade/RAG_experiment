from sentence_transformers import CrossEncoder


# -----------------------------
# Configuration
# -----------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


# -----------------------------
# Load Reranker
# -----------------------------

def load_reranker():

    print("Loading reranker model...")

    model = CrossEncoder(
        RERANKER_MODEL,
        activation_fn=None
    )

    return model


# -----------------------------
# Rerank Documents
# -----------------------------

def rerank(query, results, model):

    pairs = [
        [query, result["content"]]
        for result in results
    ]

    scores = model.predict(pairs)

    reranked_results = []

    for result, score in zip(results, scores):

        reranked_result = result.copy()

        reranked_result["reranker_score"] = float(score)

        reranked_results.append(reranked_result)

    reranked_results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True
    )

    return reranked_results