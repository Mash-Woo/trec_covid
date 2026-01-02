from typing import List, Type

import pandas as pd
import weaviate.classes as wvc
from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings
from tqdm import tqdm
from weaviate import WeaviateClient
from weaviate.collections import Collection
from weaviate.classes.config import Configure, Property, VectorDistances, DataType, Tokenization

class WeaviateDataLoader:
    def __init__(self, client: WeaviateClient, embed_model: Embeddings):
        self.client = client
        self.embed_model = embed_model

    @staticmethod
    def to_weaviate_datatype(dtype: Type) -> DataType:
        """
        Mapping Python data types to Weaviate data types.
        """
        mapping = {
            str: DataType.TEXT,
            int: DataType.INT,
            float: DataType.NUMBER,
            bool: DataType.BOOL,
            # ... Add more mappings as needed
        }

        # Default to TEXT if type not found
        
        return mapping.get(dtype, DataType.TEXT)

    def init_collection(self, collection_name: str, schema: dict, config: dict={}) -> Collection:
        """
        Initialize a Weaviate collection with the given schema.
        Args:
            collection_name (str): The name of the collection to initialize.
            schema (dict): The schema for the collection, which keys are "properties" and values are Python data types.
            e.g. schema = {
                "content": str,
                "cord_uid": str,
                "title": str,
                ...
            }
            config (dict): Additional configuration for the collection.
        Returns:
            Collection: An instance of the Collection class representing the initialized collection.
         """
        
        existing_collections = [col_name.lower() for col_name in self.client.collections.list_all().keys()]
        if collection_name.lower() in existing_collections:
            human_input = input(f"Collection '{collection_name}' already exists. Do you want to overwrite it? ([y]/n): ")
            if human_input.lower() in ["", "y", "yes"]:
                self.client.collections.delete(collection_name)
            else:
                return self.client.collections.get(collection_name)
    
        # Parse schema into Weaviate format (properties)
        weaviate_properties = [
            Property(name=key, data_type=self.to_weaviate_datatype(value))
            for key, value in schema.items()
        ]

        print(f"Creating collection '{collection_name}' with properties:\n{schema}")
        collection = self.client.collections.create(
            name=collection_name,
            properties=weaviate_properties,
            inverted_index_config=Configure.inverted_index(
                index_null_state=True,
                index_property_length=True,
                index_timestamps=True,
                stopwords_removals=[],
                bm25_k1=config["bm25_k1"] if "bm25_k1" in config else 1.2,
                bm25_b=config["bm25_b"] if "bm25_b" in config else 0.75,
            )
        )

        return collection

    def insert_data(self, collection: Collection, documents: List[Document], batch_size: int = 128) -> None:
        """
        Insert documents into Weaviate using batch embedding and Weaviate's dynamic batching.
        Args:
            collection (Collection): The Weaviate collection to insert data into.
            documents (List[Document]): A list of Document objects to be inserted.
            batch_size (int): The number of documents to process in each batch. Default is 128.
        Returns:
            None
        """
        print(f"Start insert {len(documents)} documents into collection '{collection.name}'...")

        with collection.batch.dynamic() as batch:
            # 1. Generate batches of documents
            def doc_generator(docs: List[Document], bs: int):
                for i in range(0, len(docs), bs):
                    yield docs[i : i + bs]

            # 2. Iterate over each batch of documents
            for doc_batch in tqdm(
                doc_generator(documents, batch_size), 
                total=(len(documents) + batch_size - 1) // batch_size, 
                desc="Embedding & Inserting Batches"):
                
                # 3. Get the text content of the batch
                text_batch = [doc.page_content for doc in doc_batch]
                
                # 4. Embed the batch of texts
                try:
                    vector_batch = self.embed_model.embed_documents(text_batch)
                except Exception as e:
                    print(f"[-] Error: {e}. Skipping this batch.")
                    continue
                    
                # 5. Insert each document and its vector into Weaviate
                for doc, vector in zip(doc_batch, vector_batch):
                    
                    data_object = {
                        "content": doc.page_content,
                        **doc.metadata  # Add all metadata (e.g., cord_uid, title...)
                    }
                    
                    batch.add_object(
                        properties=data_object,
                        vector=vector
                    )
        
        # The `with` block ends, Weaviate will automatically send the last batch (if any)
        print(f"Data insertion complete.")
        print(f"Tổng số objects trong collection '{collection.name}': {collection.aggregate.over_all(total_count=True).total_count}")

    def insert_data_no_embedding(self, collection: Collection, documents: List[Document], batch_size: int = 128) -> None:
        """
        Insert documents into Weaviate without generating embeddings.
        
        Args:
            collection (Collection): The Weaviate collection to insert data into.
            documents (List[Document]): A list of Document objects to be inserted.
            batch_size (int): The number of documents to process in each batch. Default is 128.
        Returns:
            None
        """
        print(f"Inserting {len(documents)} documents into collection '{collection.name}' (without embeddings)...")
        
        # Sử dụng Batch Insert để đẩy nhanh
        with collection.batch.dynamic() as batch:
            for doc_batch in tqdm(
                self._doc_generator(documents, bs=batch_size),
                total=(len(documents) + batch_size - 1) // batch_size,
                desc="Inserting Batches"
            ):
                for doc in doc_batch:
                    data_object = {
                        "content": doc.page_content,
                        **doc.metadata  # Add all metadata (e.g., cord_uid, title, abstract...)
                    }
                    batch.add_object(properties=data_object)
                
        print("Data ingestion complete!")
        print(f"Total objects in collection '{collection.name}': {collection.aggregate.over_all(total_count=True).total_count}")

    def _doc_generator(self, docs: List[Document], bs: int):
        for i in range(0, len(docs), bs):
            yield docs[i : i + bs]