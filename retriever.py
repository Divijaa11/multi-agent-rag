from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PATH,MODEL_NAME

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
)

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def retrieve_contexts_with_relevance(query: str) -> str:
    print("Query: ", query)

    results = db.similarity_search_with_relevance_scores(query)

    if not results:
        return "No relevant documents found."

    for doc, score in results:
        print(f"Text: {doc.page_content}")
        print(f"Raw Score: {score:.4f}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}\n")

    raw_scores = [score for _, score in results]

    min_score = min(raw_scores)
    max_score = max(raw_scores)

    if max_score > min_score:
        normalized_results = [
            (doc, (score - min_score) / (max_score - min_score)) for doc, score in results
        ]
    else:
        normalized_results = [(doc, 0.6) for doc, _ in results]

    for doc, norm_score in normalized_results:
        print(f"Normalized Score: {norm_score:.4f}")

    return "\n".join(
        f"Text: {doc.page_content}\n"
        f"Score: {score:.4f}\n"
        f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}\n"
        f"{'-'*80}"
        for doc, score in normalized_results if score >= 0.5 
    )
