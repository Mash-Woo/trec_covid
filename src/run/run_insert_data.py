import os
import sys
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain_core.documents import Document

__root__ = os.getcwd()
sys.path.insert(0, __root__)

from src.components.data_loader import WeaviateDataLoader
from src.components.weaviate_conn import connect_weaviate_local

load_dotenv()


# ==========================================
# DATA PROCESSING FUNCTIONS
# ==========================================

def process_dataframe_chunk(df_chunk: pd.DataFrame, content_format: str = "default") -> List[Document]:
    """
    Process a DataFrame chunk into a list of Documents.
    
    Args:
        df_chunk (pd.DataFrame): A chunk of the DataFrame to process.
        content_format (str): Format for page_content. Options:
            - "default": "Title: {title}\nAbstract: {abstract}"
            - "simple": "{title}\n{abstract}"
            - "title_only": "{title}"
    
    Returns:
        List[Document]: A list of Document objects created from the DataFrame chunk.
    """
    try:
        metadata_list = df_chunk.to_dict('records')
        
        documents_chunk = []
        for meta in metadata_list:
            title = meta.pop('title', '')
            abstract = meta.pop('abstract', '')
            
            # Format page_content based on content_format
            if content_format == "default":
                page_content = f"Title: {title}\nAbstract: {abstract}"
            elif content_format == "simple":
                page_content = f"{title}\n{abstract}"
            elif content_format == "title_only":
                page_content = title
            else:
                page_content = f"{title}\n{abstract}"
            
            documents_chunk.append(
                Document(page_content=page_content, metadata=meta)
            )
        
        return documents_chunk
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return []


def process_df_multiprocessing(df: pd.DataFrame, content_format: str = "default") -> List[Document]:
    """
    Convert DataFrame to List[Document] using multiprocessing for speed.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to process.
        content_format (str): Format for page_content.
    
    Returns:
        List[Document]: A list of Document objects created from the DataFrame.
    """
    print("Converting DataFrame to List[Document] using multiprocessing...")
    
    # Get number of CPU cores, minus one to avoid system overload
    num_workers = os.cpu_count()
    max_workers = num_workers - 1 if num_workers and num_workers > 1 else 1 
    print(f"Using {max_workers} worker processes.")

    # Split the DataFrame into chunks for processing
    df_chunks = np.array_split(df, max_workers * 2)

    documents = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit df chunks to worker processes with content_format
        futures = [
            executor.submit(process_dataframe_chunk, chunk, content_format)  # type: ignore
            for chunk in df_chunks
        ]
        
        # Receive results (with tqdm progress bar)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            documents.extend(future.result())

    print(f"Conversion complete. Total documents: {len(documents)}")
    return documents


# ==========================================
# DATA INGESTION PIPELINE
# ==========================================

def run_data_ingestion_pipeline(
    collection_name: str,
    data_path: str,
    schema: Dict[str, type],
    columns: List[str],
    use_embeddings: bool = False,
    embedding_model: Optional[str] = "text-embedding-3-small",
    batch_size: int = 128,
    content_format: str = "default",
    limit: Optional[int] = None
):
    """
    Unified data ingestion pipeline for Weaviate.
    
    Args:
        collection_name: Name of the Weaviate collection
        data_path: Path to CSV data file
        schema: Schema definition (e.g., {"cord_uid": str, "title": str, ...})
        columns: List of columns to load from CSV
        use_embeddings: Whether to generate embeddings during ingestion
        embedding_model: OpenAI embedding model name (only used if use_embeddings=True)
        batch_size: Batch size for insertion (only used if use_embeddings=True)
        content_format: Format for Document page_content
        limit: Limit number of documents for testing (None = all)
    """
    print("=" * 60)
    print(f"DATA INGESTION PIPELINE")
    print(f"Collection: {collection_name}")
    print(f"Embeddings: {'Enabled' if use_embeddings else 'Disabled'}")
    if use_embeddings:
        print(f"Embedding Model: {embedding_model}")
        print(f"Batch Size: {batch_size}")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.replace({pd.NA: None, np.nan: None})  # type: ignore
    df = df[columns]  # Select only necessary columns
    print(f"Data loaded: {len(df)} records")

    # Convert to Documents
    documents = process_df_multiprocessing(df, content_format=content_format)

    # Limit for testing
    if limit:
        print(f"Limiting to {limit} documents for testing...")
        documents = documents[:limit]

    # Connect to Weaviate
    print("\nConnecting to Weaviate...")
    weaviate_client = connect_weaviate_local()

    # Initialize embedding model if needed
    embed_model = None
    if use_embeddings:
        from langchain_openai.embeddings import OpenAIEmbeddings
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        embed_model = OpenAIEmbeddings(
            model=embedding_model, # type: ignore
            api_key=OPENAI_API_KEY,  # type: ignore
        )
        print(f"Embedding model initialized: {embedding_model}")

    # Initialize data loader
    wv_data_loader = WeaviateDataLoader(
        client=weaviate_client,
        embed_model=embed_model  # type: ignore
    )

    # Initialize collection
    collection = wv_data_loader.init_collection(
        collection_name=collection_name,
        schema=schema
    )

    # Insert data
    print(f"\nInserting {len(documents)} documents into Weaviate...")
    if use_embeddings:
        wv_data_loader.insert_data(
            collection=collection,
            documents=documents,
            batch_size=batch_size
        )
    else:
        wv_data_loader.insert_data_no_embedding(
            collection=collection,
            documents=documents
        )

    print("\n✓ Data ingestion complete!")


def main():
    # ==========================================
    # CONFIGURATION
    # ==========================================
    CONFIG = {
        "collection_name": "TREC_COVID_OpenAIEmbed_small",
        "data_path": os.path.join(__root__, "data", "raw", "CORD_19", "metadata.csv"),
        
        # Schema definition
        "schema": {
            "content": str,      # Required for embeddings
            "cord_uid": str,
            "title": str,
            "abstract": str,
        },
        
        # Columns to load from CSV
        "columns": ["cord_uid", "title", "abstract"],
        
        # Embedding options
        "use_embeddings": True,              # Set False for fast ingestion without vectors
        "embedding_model": "text-embedding-3-small",
        "batch_size": 128,                   # Only used when use_embeddings=True
        
        # Document format
        "content_format": "simple",          # Options: "default", "simple", "title_only"
        
        # Testing
        "limit": None,                       # Set to number (e.g., 1000) to limit docs for testing
    }
    # ==========================================

    run_data_ingestion_pipeline(**CONFIG)


if __name__ == "__main__":
    main()