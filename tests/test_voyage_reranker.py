import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_voyageai.rerank import VoyageAIRerank


load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

dummy_documents = [
    Document(page_content="The 2024 Apple conference focused on AI integrations.", metadata={"source": "article_1"}),
    Document(page_content="Google announced new AI models at I/O.", metadata={"source": "article_2"}),
    Document(page_content="Apple's new product line was introduced in Q3.", metadata={"source": "article_3"}),
    Document(page_content="Voyage AI rerankers achieve state-of-the-art results.", metadata={"source": "article_4"})
]

query = "What were the main topics at the recent Apple conference?"

reranker = VoyageAIRerank(
    voyage_api_key=VOYAGE_API_KEY, # type: ignore
    model="rerank-2.5",
    top_k=2
)

reranked_results = reranker._rerank(
    documents=dummy_documents,
    query=query
)

# Print results
# Build lookup table: content -> Document
doc_lookup = {
    doc.page_content: doc for doc in dummy_documents
}

print("=== Reranked Results ===")
for i, r in enumerate(reranked_results.results, 1):
    doc_index = r.index          # 🔥 index gốc
    score = r.relevance_score
    doc = dummy_documents[doc_index]

    print(f"\nRank {i}")
    print("Score:", score)
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)