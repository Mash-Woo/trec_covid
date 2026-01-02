import os
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
import numpy as np
import pandas as pd

__root__ = os.getcwd()
sys.path.insert(0, __root__)

from src.components.data_loader import WeaviateDataLoader
from src.components.weaviate_conn import connect_weaviate_local
from src.utils import read_file


load_dotenv()



# -----------------------------------------------------------------
# BƯỚC 1: ĐỊNH NGHĨA WORKER FUNCTION (Ở BÊN NGOÀI)
# -----------------------------------------------------------------
def process_dataframe_chunk(df_chunk: pd.DataFrame) -> List[Document]:
    """
    Hàm này sẽ được chạy trên một tiến trình CPU riêng biệt.
    Nó nhận vào một "mảnh" DataFrame và trả về một list Document.
    """
    try:
        # Dùng .to_dict('records') là cách nhanh nhất
        metadata_list = df_chunk.to_dict('records')
        
        documents_chunk = [
            Document(
                page_content=f"Title: {meta.pop('title', '')}\nAbstract: {meta.pop('abstract', '')}",
                metadata=meta
            )
            for meta in metadata_list
        ]
        return documents_chunk
    except Exception as e:
        print(f"Lỗi khi xử lý chunk: {e}")
        return []



# -----------------------------------------------------------------
# BƯỚC 2: SỬA LẠI HÀM MAIN
# -----------------------------------------------------------------
def run_load_data_to_weaviate():
    DATA_PATH = os.path.join(__root__, "data", "raw", "CORD_19", "metadata.csv")
    
    df = read_file(filepath=DATA_PATH)
    df = df.replace({np.nan: None}) # type: ignore
    df = df[["cord_uid", "title", "abstract"]]  # Chỉ lấy các cột cần thiết

    print(f"Data loaded from {DATA_PATH}, number of records: {len(df)}")

    # --- SỬA TỪ ĐÂY (DÙNG ĐA TIẾN TRÌNH) ---
    print("Converting DataFrame to List[Document] using multiprocessing...")
    
    # Lấy số nhân CPU, trừ đi 1 để hệ thống không bị "đứng"
    num_workers = os.cpu_count()
    max_workers = num_workers - 1 if num_workers and num_workers > 1 else 1 
    print(f"Using {max_workers} worker processes.")

    # Chia DataFrame lớn (df) thành nhiều mảnh (chunks)
    df_chunks = np.array_split(df, max_workers * 2) # Chia thành nhiều mảnh nhỏ

    documents = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Gửi các "mảnh" df cho các tiến trình xử lý
        futures = [executor.submit(process_dataframe_chunk, chunk) for chunk in df_chunks] # type: ignore
        
        # Nhận lại kết quả (có thanh tiến trình tqdm)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            documents.extend(future.result()) # Gom kết quả từ các tiến trình

    print(f"Conversion complete. Total documents processed: {len(documents)}")
    # --- SỬA ĐẾN ĐÂY ---


    # Limit documents for testing
    # documents = documents[:1000]


    client = connect_weaviate_local()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    from langchain_openai.embeddings import OpenAIEmbeddings
    openai_embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY, # type: ignore
    )

    # Load documents to Weaviate
    wv_data_loader = WeaviateDataLoader(
        client=client,
        embed_model=openai_embedding_model, # type: ignore
    )


    COLLECTION_NAME = "TREC_COVID_OpenAIEmbed_small"
    collection = wv_data_loader.init_collection(
        collection_name=COLLECTION_NAME,
        schema={
            "content": str,
            "title": str,
            "abstract": str,
        }
    )

    # documents = documents[:50]  # Giới hạn để test

    wv_data_loader.insert_data(
        collection=collection,
        documents=documents, batch_size=128
    )

if __name__ == "__main__":
    run_load_data_to_weaviate()