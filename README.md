flowchart TD
    %% --- PHASE 1: OFFLINE (DATA PREP) ---
    subgraph Offline_Process ["Phase 1: Data Preparation (Offline)"]
        direction TB
        RawData[("CORD-19 Dataset<br/>(Metadata + JSONs)")]
        
        subgraph Preprocessing ["Preprocessing & Chunking"]
            Step1["Clean Text"]
            Step2["Chunking Strategy:<br/>Paragraph + Context"]
            JSONL[("Normalized Corpus<br/>(corpus.jsonl)")]
        end

        %% Indexing Branches
        Pyserini["Pyserini Indexer"]
        Index_BM25[("BM25 Index<br/>(Lucene Folder)")]
        
        SciBERT["SciBERT Model"]
        Vector_Sci["Vectors (.npy)"]
        Index_SciBERT[("FAISS Index<br/>(SciBERT)")]
        
        BGE["BGE-M3 Model"]
        Index_BGE[("FAISS / ChromaDB<br/>(BGE-M3)")]
    end

    %% --- PHASE 2: ONLINE (SERVING) ---
    subgraph Online_Serving ["Phase 2: Search System (Online)"]
        direction TB
        Client("User / Client Interface")
        FastAPI["FastAPI Server"]

        subgraph Retrieval_Layer ["Layer 1: Parallel Retrieval (Recall)"]
            Ret_BM25[["BM25 Retriever"]]
            Ret_SciBERT[["SciBERT Retriever"]]
            Ret_BGE[["BGE-M3 Retriever"]]
        end

        subgraph Fusion_Layer ["Layer 2: Aggregation"]
            RRF("RRF Fusion Algorithm")
            Candidates["Combined Candidate List"]
        end

        subgraph Reranking_Layer ["Layer 3: Precision (Reranking)"]
            CrossEnc["Cross-Encoder<br/>BGE-Reranker"]
            SortedList["Final Ranked List"]
        end
    end

    %% --- DATA FLOW ---
    RawData --> Step1 --> Step2 --> JSONL
    JSONL --> Pyserini & SciBERT & BGE
    
    Pyserini --> Index_BM25
    SciBERT --> Vector_Sci --> Index_SciBERT
    BGE --> Index_BGE
    
    %% Online Connections
    Client -->|"1. Search Query"| FastAPI
    
    Index_BM25 -.-> Ret_BM25
    Index_SciBERT -.-> Ret_SciBERT
    Index_BGE -.-> Ret_BGE
    
    FastAPI --> Ret_BM25 & Ret_SciBERT & Ret_BGE
    
    Ret_BM25 & Ret_SciBERT & Ret_BGE -->|"Top 100"| RRF
    
    RRF -->|"Top 50-80"| Candidates
    Candidates --> CrossEnc
    CrossEnc -->|"Re-scoring"| SortedList
    SortedList -->|"Top 10 Best"| FastAPI
    FastAPI -->|"2. Return Response"| Client

    %% --- STYLING ---
    classDef storage fill:#fff,stroke:#333,stroke-dasharray: 5 5;
    classDef process fill:#e1f5fe,stroke:#01579b;
    classDef model fill:#f3e5f5,stroke:#4a148c;
    
    class RawData,JSONL,Index_BM25,Index_SciBERT,Index_BGE storage;
    class Step1,Step2,RRF,Candidates,SortedList process;
    class SciBERT,BGE,CrossEnc model;