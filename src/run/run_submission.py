import os
import sys
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from weaviate.classes.query import MetadataQuery
from weaviate.collections.classes.internal import QueryReturn
from weaviate.collections import Collection
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_voyageai.rerank import VoyageAIRerank
from langchain_core.documents import Document

__root__ = os.getcwd()
sys.path.insert(0, __root__)

from src.utils import load_topic_file, generate_submission
from src.components.weaviate_conn import connect_weaviate_local

load_dotenv()


# ==========================================
# SEARCH STRATEGIES
# ==========================================

def search_bm25(collection: Collection, query_text: str, limit: int, **kwargs) -> QueryReturn:
    """
    BM25 keyword-based search
    Args:
        collection (Collection): Weaviate collection to search
        query_text (str): The search query text
        limit (int): Number of results to return
    Returns:
        QueryReturn: Search results from Weaviate
    """
    return collection.query.bm25(
        query=query_text,
        limit=limit,
        return_metadata=MetadataQuery(score=True)
    )


def search_vector(collection: Collection, query_text: str, query_vector: List[float], limit: int, **kwargs) -> QueryReturn:
    """
    Pure vector similarity search
    Args:
        collection (Collection): Weaviate collection to search
        query_text (str): The search query text
        query_vector (List[float]): The query vector for similarity search
        limit (int): Number of results to return
    Returns:
        QueryReturn: Search results from Weaviate
    """
    return collection.query.near_vector(
        near_vector=query_vector,
        limit=limit,
        return_metadata=MetadataQuery(distance=True)
    )


def search_hybrid(collection: Collection, query_text: str, query_vector: List[float], limit: int, alpha: float = 0.5, **kwargs) -> QueryReturn:
    """
    Hybrid search combining BM25 and vector search
    Args:
        collection (Collection): Weaviate collection to search
        query_text (str): The search query text
        query_vector (List[float]): The query vector for similarity search
        limit (int): Number of results to return
        alpha (float): Weighting factor between BM25 and vector search (0.0=BM25, 1.0=vector)
    Returns:
        QueryReturn: Search results from Weaviate
    """
    return collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        alpha=alpha,  # 0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced
        limit=limit,
        return_metadata=MetadataQuery(score=True)
    )


# Mapping search method names to functions
SEARCH_METHODS = {
    "bm25": search_bm25,
    "vector": search_vector,
    "hybrid": search_hybrid,
}


# ==========================================
# MAIN PIPELINE
# ==========================================

def run_retrieval_pipeline(
    search_method: str = "hybrid",
    alpha: float = 0.5,
    top_k: int = 100,
    top_k_rerank: int = 20,
    use_reranker: bool = True,
    collection_name: str = "TREC_COVID_OpenAIEmbed_small"
):
    """
    Run the retrieval pipeline with configurable search method.
    Args:
        search_method: One of ["bm25", "vector", "hybrid"]
        alpha: Alpha parameter for hybrid search (0.0=BM25, 1.0=vector)
        top_k: Number of documents to retrieve in first stage
        top_k_rerank: Number of documents after reranking
        use_reranker: Whether to use Voyage AI reranker
        collection_name: Name of Weaviate collection
    """
    print(f"=== Retrieval Pipeline ===")
    print(f"Search Method: {search_method}")
    if search_method == "hybrid":
        print(f"Alpha: {alpha}")
    print(f"Top-K: {top_k}")
    print(f"Reranker: {'Enabled' if use_reranker else 'Disabled'} (Top-{top_k_rerank})")
    print("=" * 50)

    # Validate search method
    if search_method not in SEARCH_METHODS:
        raise ValueError(f"Invalid search method: {search_method}. Choose from {list(SEARCH_METHODS.keys())}")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

    wv_client = connect_weaviate_local()
    collection = wv_client.collections.get(collection_name)

    openai_embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY, # type: ignore
    )

    reranker = None
    if use_reranker:
        reranker = VoyageAIRerank(
            voyage_api_key=VOYAGE_API_KEY, # type: ignore
            model="rerank-2.5",
            top_k=top_k_rerank
        )

    # Print collection stats
    print(f"Total objects in collection '{collection.name}': {collection.aggregate.over_all(total_count=True).total_count}")

    topics_df = load_topic_file()
    results = []

    # Select search function
    search_fn = SEARCH_METHODS[search_method]

    for _, row in tqdm(topics_df.iterrows(), total=len(topics_df), desc="Processing queries"):
        topic_id = row['topic-id']
        query_text = f"{row['query']} {row['question']}"
        
        # Prepare search parameters
        search_params = {
            "collection": collection,
            "query_text": query_text,
            "limit": top_k,
        }
        
        # Add vector if needed
        if search_method in ["vector", "hybrid"]:
            query_vector = openai_embedding_model.embed_query(query_text)
            search_params["query_vector"] = query_vector
        
        # Add alpha if hybrid
        if search_method == "hybrid":
            search_params["alpha"] = alpha

        # Execute search
        response: QueryReturn = search_fn(**search_params)

        # Process results
        if use_reranker and reranker:
            # Prepare candidates for reranking
            rerank_query = f"{row['query']} {row['question']} {row['narrative']}"
            candidates = []
            doc_map = {}

            for idx, obj in enumerate(response.objects):
                doc_text = f"{obj.properties['title']} {obj.properties['abstract']}"
                candidates.append(Document(page_content=doc_text, metadata=obj.properties))
                doc_map[idx] = obj.properties['cord_uid']

            if candidates:
                reranked_results = reranker._rerank(
                    documents=candidates,
                    query=rerank_query
                )

                # Map reranked indices back to cord_uid
                for ranked_doc in reranked_results.results:
                    results.append({
                        "topic-id": topic_id,
                        "cord-id": doc_map[ranked_doc.index]
                    })
        else:
            # No reranking - use search results directly
            for obj in response.objects:
                results.append({
                    "topic-id": topic_id,
                    "cord-id": obj.properties['cord_uid']
                })

    results_df = pd.DataFrame(results)
    generate_submission(results_df)
    print(f"\n✓ Submission file generated successfully!")


def main():
    # ==========================================
    # CONFIGURATION
    # ==========================================
    CONFIG = {
        "search_method": "hybrid",  # Options: "bm25", "vector", "hybrid"
        "alpha": 0.5,               # For hybrid: 0.0=BM25, 1.0=vector, 0.5=balanced
        "top_k": 100,               # Number of documents to retrieve
        "top_k_rerank": 20,         # Number of documents after reranking
        "use_reranker": True,       # Enable/disable Voyage AI reranker
        "collection_name": "TREC_COVID_OpenAIEmbed_small"
    }
    # ==========================================

    run_retrieval_pipeline(**CONFIG)


if __name__ == "__main__":
    main()