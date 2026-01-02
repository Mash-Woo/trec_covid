import os
import sys

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

__root__ = os.getcwd()
sys.path.insert(0, __root__)

load_dotenv()

from src.components.data_loader import WeaviateDataLoader
from components.weaviate_conn import connect_weaviate_local

def test_weaviate_data_loader():

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    client = connect_weaviate_local()
    openai_embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=OPENAI_API_KEY, # type: ignore
    )

    wv_data_loader = WeaviateDataLoader(
        client=client,
        embed_model=openai_embedding_model,
    )

    # Test initialization
    TEST_COLLECTION_NAME = "TestCollection"
    collection = wv_data_loader.init_collection(
        collection_name=TEST_COLLECTION_NAME,
        schema={
            "content": "str",
            "number": "int",
        }
    )

    print(f"Initialized collection: {collection}")


if __name__ == "__main__":
    test_weaviate_data_loader()