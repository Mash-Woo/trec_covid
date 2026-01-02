from components.weaviate_conn import connect_weaviate_local

def test_connect_weaviate_local():
    client = connect_weaviate_local()
    
    if client:
        print(f"Connected to Weaviate local instance: {client}")
    
if __name__ == "__main__":
    test_connect_weaviate_local()