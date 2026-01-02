import os
import sys
import warnings

import weaviate
from weaviate import WeaviateClient

__root__ = os.getcwd()
sys.path.insert(0, __root__)
warnings.filterwarnings("ignore")

from src.utils import load_config

yaml_config = load_config()

def connect_weaviate_local() -> WeaviateClient:
    """
    Load Weaviate client for local instance based on yaml config settings.
    Returns:
        WeaviateClient: An instance of the Weaviate client connected to the local Weaviate server.
    """
    try:
        client = weaviate.connect_to_local(
            host=yaml_config["weaviate"]["local"]["http_host"],
            port=yaml_config["weaviate"]["local"]["http_port"],
            grpc_port=yaml_config["weaviate"]["local"]["grpc_port"],
        )

        return client
    
    except Exception as e:
        raise ConnectionError(f"[-] Failed to connect to local Weaviate instance: {e}")