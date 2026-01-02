# TREC-COVID Information Retrieval (Round 3)

Hệ thống tìm kiếm thông tin y sinh sử dụng nhằm hỗ trợ nghiên cứu COVID-19.

## Problem Statement

Trong bối cảnh đại dịch COVID-19, các nhà nghiên cứu, bác sĩ cần tìm kiếm thông tin đáng tin cậy về virus và tác động của nó. Việc tìm kiếm thông tin nhanh, chính xác và nhanh chóng từ hàng nghìn tài liệu nghiên cứu y sinh phân mảnh là một thách thức lớn.

TREC-COVID Challenge được tổ chức bởi Allen Institute for AI (AI2), NIST, NLM, OHSU và UTHealth nhằm:
- Xây dựng hệ thống tìm kiếm thông tin (Information Retrieval) hiệu quả cho các tài liệu liên quan đến COVID-19
- Đánh giá các phương pháp Information Retrieval khác nhau trên bộ dữ liệu CORD-19

## Objectives

### Mục tiêu chính
1. **Xây dựng hệ thống retrieval**: Phát triển hệ thống có khả năng trả về danh sách các tài liệu được xếp hạng (ranked documents) từ bộ dữ liệu CORD-19 cho 40 topics/queries liên quan đến COVID-19 *(35 topics từ Round 1&2 + 5 topics mới từ Round 3)*.

2. **Đánh giá hiệu suất**: Sử dụng NDCG (Normalized Discounted Cumulative Gain) để đo lường chất lượng ranking của các tài liệu trả về.

3. **So sánh các phương pháp**: Thử nghiệm và đánh giá hiệu quả của các phương pháp khác nhau:
   - BM25 (keyword-based search)
   - Vector search (semantic similarity)
   - Hybrid search (kết hợp BM25 và vector)
   - Reranking (sử dụng mô hình học máy để cải thiện ranking)
   - Các kỹ thuật tiền xử lý dữ liệu và mở rộng truy vấn (query expansion)
   - Ensemble methods (kết hợp nhiều mô hình)
   - Fine-tuning + transfer learning trên các mô hình ngôn ngữ lớn (large language models)

### Yêu cầu kỹ thuật
- Trả về ranked list của documents từ CORD-19 dataset
- Xử lý 40 topics *(35 topics từ Round 1&2 + 5 topics mới từ Round 3)*
- Loại bỏ các documents đã được đánh giá trong các round trước
- Định dạng submission: `topicid,docid` với mỗi dòng là một cặp topic-document

### Relevance Judgments
Các tài liệu được đánh giá theo thang điểm 3 mức:
- **Relevant (2)**: Tài liệu trả lời đầy đủ câu hỏi trong topic
- **Partially Relevant (1)**: Tài liệu trả lời một phần câu hỏi
- **Not Relevant (0)**: Không liên quan

## Dataset

### CORD-19 Dataset
- **Nguồn**: COVID-19 Open Research Dataset (CORD-19)
- **Phiên bản**: May 19, 2020
- **Nội dung**: Bộ sưu tập các bài báo nghiên cứu y sinh về COVID-19
- **Định dạng**: JSON (pdf_json, pmc_json)

### Topics Structure
Mỗi topic bao gồm:
- **topic-id**: ID của topic (1-40)
- **query**: Câu truy vấn ngắn gọn
- **question**: Câu hỏi chi tiết
- **narrative**: Mô tả đầy đủ về information need

## Installation

### 1. Prerequisites
- Python 3.10+ (Recommended: 3.11.13)
- Docker và Docker Compose
- API Keys: OpenAI, Voyage AI (optional, cho vector search và reranking)

### 2. Clone Repository
```bash
git clone https://github.com/dtruong46me/trec-covid-ir-3.git
cd trec-covid-ir-3
```

### 3. Cài đặt Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Weaviate Vector Database

#### 4.1. Start Weaviate với Docker Compose
```bash
docker-compose up -d
```

File `docker-compose.yaml` đã được cấu hình sẵn với:
- Weaviate server tại `http://localhost:8080`
- Các modules: `text2vec-openai`, `generative-openai`

#### 4.2. Kiểm tra Weaviate đã chạy
```bash
docker ps | grep weaviate
```

Nếu trả về container Weaviate, nghĩa là đã chạy thành công.

### 5. Cấu hình API Keys

Tạo file `.env` trong thư mục root:
```bash
# OpenAI API (cho embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# Voyage AI API (cho reranking - optional)
VOYAGE_API_KEY=your_voyage_api_key_here
```

**Lấy API keys:**
- OpenAI: https://platform.openai.com/api-keys (At least 5 USD)
- Voyage AI: https://www.voyageai.com/ (Free Tier available)

### 6. Tải Dataset từ Kaggle

```bash
# Giải nén vào thư mục /data/raw/CORD_19/
data/raw/CORD_19/
```

### 7. Load Data vào Weaviate

#### 7.1. Load data KHÔNG có embeddings (nhanh hơn)
```bash
python src/run/run_insert_data.py
```

**Cấu hình trong file** (`CONFIG` dictionary):
```python
CONFIG = {
    "collection_name": "Cord19Papers",
    "data_path": ", # Đường dẫn tới metadata.csv
    "use_embeddings": False,  # Tắt embeddings
    ... # Các cấu hình khác
}
```

#### 7.2. Load data CÓ embeddings (cho vector search)
Sửa CONFIG trong file `run_insert_data.py`:
```python
CONFIG = {
    "use_embeddings": True,  # Bật embeddings
    "embedding_model": "text-embedding-3-small" # Bắt buộc phải có OpenAI API Key
    ... # Các cấu hình khác
}
```

**Lưu ý**: 
- Cần có `OPENAI_API_KEY` trong `.env`
- Quá trình này sẽ lâu hơn (tạo embeddings cho từng document)
- Cost: ~$0.0001/1K tokens với `text-embedding-3-small`

### 8. Chạy Retrieval và Generate Submission

File chính: [src/run/run_submission.py](src/run/run_submission.py)

#### 8.1. Cấu hình Retrieval

Trong file `run_submission.py`, sửa `CONFIG`:
```python
CONFIG = {
    "search_method": "bm25",      # "bm25", "vector", "hybrid"
    "alpha": 0.5,                 # Cho hybrid: 0.0=BM25, 1.0=vector
    "top_k": 1000,                # Số documents trả về
    "top_k_rerank": 100,          # Số documents sau rerank (optional)
    "use_reranker": False,        # True = sử dụng Voyage reranker
    "collection_name": "Cord19Papers"
}
```

#### 8.2. Các phương pháp Search

**1. BM25 Search (Keyword-based)**
```python
CONFIG = {
    "search_method": "bm25",
    "top_k": 100,
    "use_reranker": False
}
```
```bash
python src/run/run_submission_bm25.py
```

**2. Vector Search (Semantic similarity)**
```python
CONFIG = {
    "search_method": "vector",
    "top_k": 100,
    "use_reranker": False
}
```
*Yêu cầu: Data phải được load với embeddings*

**3. Hybrid Search (BM25 + Vector)**
```python
CONFIG = {
    "search_method": "hybrid",
    "alpha": 0.5,  # Thử nghiệm: 0.3, 0.5, 0.7
    "top_k": 100,
    "use_reranker": False
}
```

**4. Với Reranker (cải thiện accuracy)**
```python
CONFIG = {
    "search_method": "hybrid",
    "alpha": 0.5,
    "top_k": 200,
    "top_k_rerank": 20,
    "use_reranker": True  # Cần VOYAGE_API_KEY
}
```

#### 8.3. Output

File submission được tạo tại:
```
output/submissions/{timestamp}/submission.csv
```

Metrics được lưu tại:
```
output/metrics/{timestamp}.json
```

### 9. Submit lên Kaggle

Submit qua Web UI:
1. Truy cập: https://www.kaggle.com/c/trec-covid-information-retrieval/submit
2. Upload file `submission.csv`
3. Thêm description
4. Submit và chờ kết quả

## Data Directory Structure

```
data/
├── raw/                                         # Original, unprocessed datasets from TREC-COVID
│   └── CORD_19/                                 # CORD-19 corpus and TREC-COVID evaluation resources
│       ├── cord_19_embeddings_2020-05-19.csv    # Pre-computed document embeddings for semantic search
│       ├── metadata.csv                         # Document metadata (titles, authors, abstracts, DOIs, etc.)
│       ├── topics-rnd3.csv                      # Query topics for Round 3 (50 COVID-19 information needs)
│       ├── qrels.csv                            # Relevance judgments (ground truth for evaluation metrics)
│       ├── docids-rnd3.txt                      # Document IDs eligible for Round 3 submission
│       ├── submission.csv                       # Template file showing required submission format
│       ├── pdf_json/                            # Full-text papers extracted from PDFs (hash-named files)
│       │   ├── 0001418189999fea7f7cbe3e82703d71c85a6fe5.json
│       │   ├── 0003793cf9e709bc2b9d0c8111186f78fb73fc04.json
│       │   └── ...
│       └── pmc_json/                            # Full-text papers from PubMed Central (PMC ID format)
│           ├── PMC1054884.xml.json
│           ├── PMC1065028.xml.json
│           └── ...
└── processed/                                   # Preprocessed datasets (cleaned text, custom embeddings, etc.)
                                                 # Currently empty - populated during pipeline execution
```

## Project Structure

```
trec-covid-ir-3/
├── config.yaml                  # Cấu hình chung (nếu có)
├── docker-compose.yaml          # Docker compose cho Weaviate
├── README.md                    # File này
├── requirements.txt             # Python dependencies
├── .env                         # API keys (không commit)
├── .gitignore
│
├── data/                        # Xem phần "Cấu trúc Data Directory" ở trên
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_baseline_testing.ipynb
│   ├── 03_evaluation_analysis.ipynb
│   └── lib/
│       └── utils.py
│
├── src/
│   ├── utils.py                 # Utility functions
│   ├── components/
│   │   ├── data_loader.py      # WeaviateDataLoader class
│   │   └── weaviate_conn.py    # Weaviate connection
│   ├── preprocessing/           # Data preprocessing scripts
│   └── run/
│       ├── run_data_loader.py
│       ├── run_insert_data_without_embed.py  # Load data (with/without embeddings)
│       ├── run_submission_bm25.py            # Generate submission
│       └── run_evaluation.py
│
├── output/
│   ├── submissions/             # Submission files
│   │   └── {timestamp}/
│   │       └── submission.csv
│   └── metrics/                 # Evaluation metrics
│       └── {timestamp}.json
│
└── tests/                       # Unit tests
    ├── test_data_loader.py
    └── test_weaviate_conn.py
```

## Methodology

### 1. Data Ingestion Pipeline
- **Input**: CORD-19 metadata CSV với ~128K papers
- **Processing**: 
  - Multiprocessing để tăng tốc độ (ProcessPoolExecutor)
  - Flexible content format: default, simple, title_only
  - Optional embeddings generation với OpenAI `text-embedding-3-small`
- **Output**: Weaviate collection với BM25 index và optional vector index

### 2. Search Strategies

#### BM25 (Okapi BM25)
- Thuật toán keyword-based ranking
- Tham số: k1=1.2, b=0.75 (Weaviate defaults)
- Phù hợp cho: Exact term matching

#### Vector Search
- Embeddings model: OpenAI `text-embedding-3-small`
- Similarity metric: Cosine similarity
- Phù hợp cho: Semantic similarity, paraphrase queries

#### Hybrid Search
- Kết hợp BM25 và Vector search
- Alpha parameter: 
  - α = 0: Pure BM25
  - α = 0.5: Balanced
  - α = 1: Pure Vector
- Công thức: `score = α × vector_score + (1-α) × bm25_score`

### 3. Reranking (Optional)
- Model: Voyage AI `rerank-2.5`
- Input: Top-K documents từ initial search
- Output: Reranked top-N documents
- Cải thiện precision ở top results

### 4. Query Expansion (Optional)
- Mở rộng query với synonyms, related terms
- Cải thiện recall và NDCG

### 5. Ensemble Methods (Optional)
- Kết hợp multiple submissions
- Phương pháp: Rank fusion, weighted averaging

### 6. Fine-tuning + Transfer Learning (Optional)
- Sử dụng pre-trained language models (e.g., SciBERT, BGE,..)
- Fine-tune trên tập dữ liệu y sinh cho retrieval tasks

## Evaluation Metrics

### NDCG (Normalized Discounted Cumulative Gain)
- Metric chính của competition
- Đánh giá chất lượng ranking với graded relevance
- Công thức: 
  ```
  DCG@k = Σ(i=1 to k) (2^rel_i - 1) / log2(i + 1)
  NDCG@k = DCG@k / IDCG@k
  ```
- Điểm cao hơn = ranking tốt hơn

### Submission Format
```csv
topicid,docid
1,000ajevz
1,000q5l5n
1,000tfenb
...
40,zzyfmkb
```
- Mỗi topic: 1000 documents được ranked
- Total: 40,000 dòng (40 topics × 1000 docs)

## Experiments

### Recommended Experiments

1. **BM25 Baseline**
   ```python
   CONFIG = {"search_method": "bm25", "top_k": 100}
   ```

2. **Hybrid với các giá trị alpha khác nhau**
   ```python
   # Test alpha = 0.3, 0.5, 0.7
   for alpha in [0.3, 0.5, 0.7]:
       CONFIG = {"search_method": "hybrid", "alpha": alpha, "top_k": 100}
   ```

3. **Sử dụng Reranking**
   ```python
   CONFIG = {
       "search_method": "hybrid", 
       "alpha": 0.5, 
       "top_k": 200,
       "top_k_rerank": 50,
       "use_reranker": True
   }
   ```

## Troubleshooting

### Weaviate Connection Error
```bash
# Kiểm tra Weaviate đang chạy
docker ps | grep weaviate

# Restart Weaviate
docker-compose restart

# Xem logs
docker-compose logs weaviate
```

### Out of Memory
- Giảm `batch_size` trong CONFIG
- Giảm `limit` để load ít data hơn
- Tăng RAM cho Docker Desktop

### OpenAI Rate Limit
- Giảm tốc độ request (thêm sleep)
- Nâng cấp OpenAI tier
- Sử dụng pre-computed embeddings từ CORD-19

### Slow Performance
- Sử dụng `use_embeddings=False` cho baseline
- Tăng số workers trong multiprocessing
- Tối ưu hóa Weaviate resources trong docker-compose

## References

- **Competition**: [TREC-COVID on Kaggle](https://www.kaggle.com/competitions/trec-covid-information-retrieval)
- **Dataset**: [CORD-19 Dataset](https://allenai.org/data/cord-19)
- **TREC-COVID**: [Official Website](https://ir.nist.gov/trec-covid/)
- **Weaviate**: [Documentation](https://weaviate.io/developers/weaviate)
- **BM25**: [Okapi BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)

## Citation

```bibtex
@misc{trec-covid-2020,
  author = {Ellen Voorhees and Ian Soboroff and Walter Reade and Julia Elliott},
  title = {TREC-COVID Information Retrieval},
  year = {2020},
  url = {https://kaggle.com/competitions/trec-covid-information-retrieval},
  publisher = {Kaggle}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the terms specified in the LICENSE file.

## Team

- **Organization**: Allen Institute for AI (AI2), NIST, NLM, OHSU, UTHealth
- **Competition**: TREC-COVID Round 3

---

**Last Updated**: January 2026
