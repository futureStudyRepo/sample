# Milvus 벡터 데이터베이스 기초 실습 (ORM 방식)

> 핵심 개념 · 실습 코드 · 아키텍처 가이드  
> **pymilvus ORM 스타일** — `connections` / `Collection` / `utility` 기반

| 주제 | 내용 |
|------|------|
| 컬렉션 관리 | 스키마 설계, 파티션, 인덱스 생성 |
| 데이터 관리 | Dense / Sparse / 멀티벡터 삽입·수정·삭제 |
| 조회 (Query) | 스칼라 필터 기반 데이터 조회 |
| 벡터 검색 | ANN 검색, 파라미터 튜닝 |
| 인덱스 | IVF_FLAT / HNSW / DISKANN 비교 |
| 랭킹 & 필터 | Pre/Post-filter, 거리 메트릭 |
| 하이브리드 검색 | Dense + Sparse, RRF / WeightedRanker |
| 실전 구성 | RAG 파이프라인 End-to-End |

---

## 00. 공통 준비

```bash
pip install pymilvus numpy sentence-transformers
```

```python
from pymilvus import (
    connections, utility,
    Collection, CollectionSchema, FieldSchema,
    DataType, AnnSearchRequest, WeightedRanker, RRFRanker,
)
import numpy as np

DIM             = 128
COLLECTION_NAME = "docs"

# ── 연결 ─────────────────────────────────────────────────────
def connect_milvus():
    if not connections.has_connection("default"):
        connections.connect(alias="default", host="localhost", port=19530)

connect_milvus()
print(connections.get_connection_addr("default"))
```

> **NOTE** `connections.has_connection()` 체크로 중복 연결을 방지합니다.  
> FastAPI / Flask 등 함수가 반복 호출되는 환경에서 필수 패턴입니다.

---

## 01. 컬렉션 관리 (Collection Management)

### 핵심 개념

| 개념 | 설명 | RDB 대응 |
|------|------|---------|
| Collection | 데이터 저장의 최상위 단위, 스키마+인덱스 포함 | Table |
| Field | 개별 데이터 속성. 벡터/스칼라 구분 | Column |
| Entity | 하나의 데이터 레코드 | Row |
| Segment | 내부 데이터 관리 단위 (Growing/Sealed) | Partition(내부) |
| Partition | 컬렉션 내 논리 분할 (tag 기반) | Table Partition |

### Lab 1-1. 컬렉션 생성

```python
def ensure_collection():
    connect_milvus()
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id",        dtype=DataType.INT64,       is_primary=True, auto_id=False),
        FieldSchema(name="title",      dtype=DataType.VARCHAR,     max_length=256),
        FieldSchema(name="category",   dtype=DataType.VARCHAR,     max_length=64),
        FieldSchema(name="embedding",  dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, enable_dynamic_field=True)
    col    = Collection(COLLECTION_NAME, schema)

    col.create_index(
        field_name   = "embedding",
        index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 64}},
    )
    col.load()
    return col

col = ensure_collection()
print(utility.list_collections())   # ['docs']
```

### Lab 1-2. 파티션 관리

```python
# 파티션 생성
col.create_partition("tech")
col.create_partition("news")

# 파티션 목록 확인
print([p.name for p in col.partitions])   # ['_default', 'tech', 'news']

# 특정 파티션에만 검색 (성능 향상)
col.search(data=[q_vec], anns_field="embedding", param={}, limit=5,
           partition_names=["tech"])

# 파티션 삭제
col.drop_partition("news")
```

### Lab 1-3. 컬렉션 수명주기

```python
col.describe()                     # 스키마·인덱스 상세 확인
col.num_entities                   # 엔티티 수 (flush 후 정확)
col.flush()                        # 메모리 → 디스크 반영

col.release()                      # 메모리 해제 (검색 불가 상태)
col.load()                         # 메모리 로드 (검색 가능 상태)

utility.rename_collection(COLLECTION_NAME, "documents")
utility.drop_collection("documents")
```

> **NOTE** `create_index()` 후 반드시 `load()`를 호출해야 검색이 가능합니다.  
> `ensure_collection()` 패턴처럼 생성 시 load까지 묶어두면 실수를 줄일 수 있습니다.

---

## 02. 데이터 관리 (Dense / Sparse / Multi-Vector)

### 핵심 개념

| 벡터 유형 | DataType | 특징 | 대표 사용처 |
|----------|---------|------|-----------|
| Dense | FLOAT_VECTOR | 모든 차원에 값. dim=768/1536 일반적 | 문장·이미지 임베딩 |
| Sparse | SPARSE_FLOAT_VECTOR | 비제로 값만 저장, 고차원(30K+) | BM25, TF-IDF 키워드 |
| Binary | BINARY_VECTOR | 비트 단위 저장, 초고속 비교 | 해시 기반 중복 탐지 |
| Float16 | FLOAT16_VECTOR | FP16 정밀도, 메모리 절반 | GPU 임베딩 최적화 |

```
Dense  : [0.21, 0.85, 0.43, 0.67, 0.12, 0.94, ...]  ← 모든 차원에 값
Sparse : {142: 0.91, 3801: 0.54, 29103: 0.37, ...}  ← 비제로 인덱스만
```

### Lab 2-1. Dense 벡터 삽입

```python
col = Collection(COLLECTION_NAME)

data = [
    [i for i in range(1000)],                                          # id
    [f"문서 제목 {i}" for i in range(1000)],                           # title
    ["tech" if i % 2 == 0 else "news" for i in range(1000)],          # category
    [np.random.rand(DIM).tolist() for _ in range(1000)],               # embedding
]

res = col.insert(data)
col.flush()
print(f"삽입: {res.insert_count}개")
```

> **NOTE** ORM 방식에서 `insert()`는 **필드 순서대로 리스트를 담은 리스트** 형태로 전달합니다.  
> dict 리스트 형태는 `MilvusClient` 방식에서 사용합니다.

### Lab 2-2. Sparse 벡터 삽입

```python
# Sparse 컬렉션 생성
sp_fields = [
    FieldSchema(name="id",     dtype=DataType.INT64,              is_primary=True, auto_id=False),
    FieldSchema(name="text",   dtype=DataType.VARCHAR,             max_length=512),
    FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
]
sp_col = Collection("docs_sparse", CollectionSchema(sp_fields))
sp_col.create_index(
    field_name   = "sparse",
    index_params = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP",
                    "params": {"drop_ratio_build": 0.2}},
)
sp_col.load()

# Sparse 벡터: dict {term_id: weight} 형태
ids     = list(range(500))
texts   = [f"doc {i}" for i in range(500)]
sparses = [
    {j: float(np.random.rand()) for j in np.random.choice(30000, 20, replace=False)}
    for _ in range(500)
]
sp_col.insert([ids, texts, sparses])
sp_col.flush()
```

### Lab 2-3. 데이터 수정·삭제

```python
col = Collection(COLLECTION_NAME)

# ID 기반 삭제 (표현식 필수)
col.delete(expr="id in [0, 1, 2]")

# 조건 기반 삭제
col.delete(expr='category == "news" and score < 70')

# Upsert (있으면 update, 없으면 insert) — v2.3+
upsert_data = [
    [10],
    ["업데이트된 제목"],
    ["tech"],
    [np.random.rand(DIM).tolist()],
]
col.upsert(upsert_data)
col.flush()
```

> **NOTE** Milvus는 직접 update를 지원하지 않습니다. `delete → insert` 또는 `upsert`를 사용하세요.

---

## 03. 조회 (Query — 스칼라 필터 기반)

### 핵심 개념

`query()`는 벡터 유사도가 아닌 스칼라 필드 조건으로 데이터를 조회합니다. SQL의 SELECT와 동일한 개념입니다.

| 연산자 | 예시 | 설명 |
|--------|------|------|
| 비교 | `score >= 80` | 숫자 비교 |
| 문자열 | `category == "tech"` | 정확 일치 |
| 범위 | `id in [1, 2, 3]` | IN 절 |
| 논리 | `score > 70 and category == "tech"` | AND / OR / NOT |
| LIKE | `title like "Milvus%"` | 접두어 매칭 |
| JSON 필드 | `meta["lang"] == "ko"` | JSON 필드 접근 |

### Lab 3-1. 기본 Query

```python
col = Collection(COLLECTION_NAME)

rows = col.query(
    expr          = 'category == "tech"',
    output_fields = ["id", "title", "category"],
    limit         = 20,
)
for r in rows:
    print(r["id"], r["title"])
```

### Lab 3-2. ID 직접 조회 & 전체 스캔

```python
# ID 리스트로 직접 조회
rows = col.query(expr="id in [10, 20, 30]", output_fields=["id", "title"])

# 전체 데이터 스캔 (페이지네이션)
offset = 0
while True:
    batch = col.query(
        expr          = "id >= 0",
        output_fields = ["id", "title"],
        limit         = 100,
        offset        = offset,
    )
    if not batch:
        break
    print(f"  offset={offset}: {len(batch)}건")
    offset += 100
```

> **TIP** `col.query()`는 인덱스 없이도 동작하지만, 대용량에서는 필터 필드에 스칼라 인덱스를 추가하는 것을 권장합니다.

---

## 04. 벡터 검색 (ANN Search)

### 핵심 개념

```
Query Text → Embedding Model → Query Vector → ANN Search → Ranking & Filter → Top-K Results → LLM(RAG)
   입력         SBERT/OpenAI      부동소수 배열    IVF/HNSW      nprobe/ef       distance+fields    답변 생성
```

| 메트릭 | 수식 개요 | 적합한 데이터 |
|--------|---------|-------------|
| COSINE | cos(θ) = A·B / (\|A\|\|B\|) | 텍스트, 문서 임베딩 (방향성) |
| L2 | √Σ(aᵢ-bᵢ)² | 이미지, 수치 특징 (절대 거리) |
| IP | Σ aᵢ·bᵢ | 정규화 벡터에서 COSINE과 동일 |
| JACCARD | 교집합/합집합 | Binary 벡터, 집합 유사도 |

### Lab 4-1. 기본 ANN 검색

```python
col = Collection(COLLECTION_NAME)
col.load()

q_vec = np.random.rand(DIM).tolist()

results = col.search(
    data         = [q_vec],           # 배치 검색 가능: [vec1, vec2, ...]
    anns_field   = "embedding",       # 검색할 벡터 필드명
    param        = {"metric_type": "COSINE", "params": {"nprobe": 16}},
    limit        = 5,                 # Top-K
    output_fields= ["title", "category"],
)

for hit in results[0]:
    print(f"id={hit.id}  dist={hit.distance:.4f}  title={hit.entity.get('title')}")
```

> **NOTE** ORM 방식은 결과가 `SearchResult` 객체입니다.  
> `hit.id`, `hit.distance`, `hit.entity.get('필드명')` 으로 접근합니다.

### Lab 4-2. Range Search (거리 범위 제한)

```python
results = col.search(
    data       = [q_vec],
    anns_field = "embedding",
    param      = {
        "metric_type": "COSINE",
        "params": {
            "nprobe":       16,
            "radius":       0.8,   # 최소 유사도
            "range_filter": 1.0,   # 최대 유사도
        },
    },
    limit         = 20,
    output_fields = ["title"],
)
```

### Lab 4-3. Iterator 기반 대량 검색

```python
from pymilvus import SearchIterator

iterator = col.search_iterator(
    data         = [q_vec],
    anns_field   = "embedding",
    param        = {"metric_type": "COSINE", "params": {"nprobe": 8}},
    batch_size   = 100,
    output_fields= ["title"],
)
total = 0
while True:
    batch = iterator.next()
    if not batch:
        break
    total += len(batch)
iterator.close()
print(f"총 {total}건 처리")
```

---

## 05. 인덱스 (Index)

### 핵심 개념

| 인덱스 | 알고리즘 | 주요 파라미터 | 권장 규모 |
|--------|---------|-------------|---------|
| FLAT | 완전 탐색 | 없음 | < 100만 |
| IVF_FLAT | 역인덱스 클러스터 | nlist(64~2048), nprobe(검색 시) | 100만~1억 |
| IVF_SQ8 | IVF+스칼라 양자화 | nlist | 메모리 제약 |
| IVF_PQ | IVF+곱 양자화 | nlist, m, nbits | 고차원 대용량 |
| HNSW | 계층 그래프 | M(16~64), efConstruction(100~500) | 온라인 서비스 |
| DISKANN | 그래프+디스크 | search_list, build_ratio | > 1억 (디스크) |
| AUTOINDEX | Zilliz 자동 | 없음 | Zilliz Cloud |

### Lab 5-1. 인덱스 교체 (IVF_FLAT → HNSW)

```python
col = Collection(COLLECTION_NAME)

# 기존 인덱스 해제 후 삭제
col.release()
col.drop_index()

# HNSW 인덱스 생성
col.create_index(
    field_name   = "embedding",
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M":              16,   # 그래프 연결 수 (↑정확도, ↑메모리)
            "efConstruction": 200,  # 빌드 시 탐색 폭 (↑정확도, ↑빌드시간)
        },
    },
)
col.load()

# HNSW 검색 시 ef 파라미터 조정
results = col.search(
    data       = [q_vec],
    anns_field = "embedding",
    param      = {"metric_type": "COSINE", "params": {"ef": 64}},
    # ef >= limit 권장. ↑ef = ↑정확도, ↑레이턴시
    limit      = 5,
)
```

### Lab 5-2. 스칼라 필드 인덱스

```python
col = Collection(COLLECTION_NAME)

# VARCHAR 필드에 Trie 인덱스 (필터 성능 향상)
col.create_index(
    field_name   = "category",
    index_params = {"index_type": "Trie"},   # 문자열: Trie / 숫자: STL_SORT / INVERTED
)

# 인덱스 목록 확인
print(col.indexes)
```

> **TIP** HNSW는 빠른 검색과 높은 정확도로 온라인 서비스에 최적입니다.  
> 데이터가 수십억 건이고 메모리가 부족하다면 DISKANN을 선택하세요.

---

## 06. 랭킹 & 필터 (Ranking & Filtering)

### 핵심 개념

```
전체 Entity(100만) → Pre-Filter 적용(2만) → ANN 검색 nprobe=16(100개) → Re-Rank & Limit(Top-5) → Output Fields 반환
                    expr='cat=="news"'         col.search()                   distance 정렬           output_fields=[...]
```

| 방식 | 설명 | 장단점 |
|------|------|--------|
| Pre-filter (기본) | 필터 후 ANN 검색 | 결과 수 보장 어려움, 정밀 필터에 최적 |
| Post-filter | ANN 후 필터 적용 | 결과 수 예측 가능, 대용량 필터에 유리 |
| Range Search | 거리 범위로 후보 제한 | 유사도 임계값 적용 가능 |

### Lab 6-1. Pre-filter 검색

```python
col = Collection(COLLECTION_NAME)

results = col.search(
    data          = [q_vec],
    anns_field    = "embedding",
    param         = {"metric_type": "COSINE", "params": {"nprobe": 16}},
    limit         = 10,
    expr          = 'category == "tech"',   # 필터 표현식
    output_fields = ["title", "category"],
)

for hit in results[0]:
    print(hit.id, hit.distance, hit.entity.get("title"))

# IN 연산자로 다중 값 필터
results2 = col.search(
    data       = [q_vec],
    anns_field = "embedding",
    param      = {"metric_type": "COSINE", "params": {"nprobe": 8}},
    limit      = 5,
    expr       = "id in [10, 20, 30, 40, 50]",
)
```

### Lab 6-2. 거리 메트릭 비교 실습

```python
for metric in ["COSINE", "L2", "IP"]:
    res = col.search(
        data       = [q_vec],
        anns_field = "embedding",
        param      = {"metric_type": metric, "params": {"nprobe": 8}},
        limit      = 3,
        output_fields = ["title"],
    )
    ids   = [hit.id for hit in res[0]]
    dists = [round(hit.distance, 4) for hit in res[0]]
    print(f"{metric:8s} → ids={ids}  dists={dists}")

# COSINE: 높을수록 유사 (1.0 = 완벽 일치)
# L2:     낮을수록 유사 (0.0 = 완벽 일치)
# IP:     높을수록 유사 (정규화된 벡터에서)
```

### Lab 6-3. 그룹화 검색 (GroupBy)

```python
results = col.search(
    data            = [q_vec],
    anns_field      = "embedding",
    param           = {"metric_type": "COSINE", "params": {"nprobe": 16}},
    limit           = 6,
    group_by_field  = "category",   # 각 category에서 best 1개
    output_fields   = ["title", "category"],
)
for hit in results[0]:
    print(hit.entity.get("category"), hit.entity.get("title"))
```

---

## 07. 하이브리드 검색 (Hybrid Search)

### 핵심 개념

```
Query ──┬──► Dense Encoder ──► Dense Search  ──► 결과 10개 ──┐
        │    (SBERT/OpenAI)    (COSINE/L2)                    ├──► Re-Ranker ──► Top-K
        └──► Sparse Encoder ──► Sparse Search ──► 결과 10개 ──┘    (RRF/Weighted)
             (BM25/SPLADE)     (IP/BM25)
```

| Re-Ranker | 수식 | 특징 |
|-----------|------|------|
| RRFRanker | score = Σ 1/(k + rankᵢ) | 순위 기반. 점수 스케일 무관, 안정적 |
| WeightedRanker | score = Σ wᵢ × normScoreᵢ | 가중치 직접 지정. 도메인 지식 반영 |

### Lab 7-1. 하이브리드 컬렉션 구성

```python
h_fields = [
    FieldSchema(name="id",     dtype=DataType.INT64,              is_primary=True, auto_id=False),
    FieldSchema(name="text",   dtype=DataType.VARCHAR,             max_length=512),
    FieldSchema(name="dense",  dtype=DataType.FLOAT_VECTOR,        dim=DIM),
    FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
]
h_col = Collection("hybrid_docs", CollectionSchema(h_fields, enable_dynamic_field=True))

h_col.create_index("dense",  {"index_type": "HNSW",                  "metric_type": "COSINE",
                               "params": {"M": 16, "efConstruction": 200}})
h_col.create_index("sparse", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP",
                               "params": {"drop_ratio_build": 0.2}})
h_col.load()

# 데이터 삽입
ids     = list(range(500))
texts   = [f"doc {i}" for i in range(500)]
denses  = [np.random.rand(DIM).tolist() for _ in range(500)]
sparses = [
    {j: float(np.random.rand()) for j in np.random.choice(30000, 15, replace=False)}
    for _ in range(500)
]
h_col.insert([ids, texts, denses, sparses])
h_col.flush()
```

### Lab 7-2. RRFRanker 하이브리드 검색

```python
dense_req = AnnSearchRequest(
    data       = [np.random.rand(DIM).tolist()],
    anns_field = "dense",
    param      = {"metric_type": "COSINE", "params": {"ef": 100}},
    limit      = 10,
)
sparse_req = AnnSearchRequest(
    data       = [{j: float(np.random.rand()) for j in np.random.choice(30000, 10, replace=False)}],
    anns_field = "sparse",
    param      = {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
    limit      = 10,
)

# RRF: k=60 (기본값). 낮을수록 상위 랭크에 가중치 집중
results = h_col.hybrid_search(
    reqs          = [dense_req, sparse_req],
    rerank        = RRFRanker(k=60),
    limit         = 5,
    output_fields = ["text"],
)
for hit in results[0]:
    print(f"id={hit.id}  score={hit.distance:.4f}  text={hit.entity.get('text')}")
```

### Lab 7-3. WeightedRanker (가중치 조정)

```python
# dense 70%, sparse 30% 가중치
results_w = h_col.hybrid_search(
    reqs   = [dense_req, sparse_req],
    rerank = WeightedRanker(0.7, 0.3),   # 순서 = reqs 순서와 일치
    limit  = 5,
)

# 실험: 가중치를 변경하며 결과 비교
for w in [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)]:
    r = h_col.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=WeightedRanker(*w),
        limit=3,
    )
    ids = [hit.id for hit in r[0]]
    print(f"dense={w[0]} sparse={w[1]} → {ids}")
```

> **TIPS** 키워드 정확성이 중요한 법률·의학 도메인은 sparse 가중치를,  
> 의미적 유사성이 중요한 추천·QA는 dense 가중치를 높이세요.

---

## 08. 실전 구성 — RAG 파이프라인

### 아키텍처

```
[ Ingest Pipeline ]
문서 원본 → Chunking → Embedding Model → Milvus Insert
(PDF/HTML)  (Splitter)  (SBERT/OpenAI)   (col.insert())
                                                │
                                           저장된 벡터
                                                │
[ Query Pipeline ]                              ▼
User Query → Embedding → col.search() → Re-Rank & Filter → LLM → Answer
  (질문)      (벡터화)    (Dense+Sparse)   (RRF/메타필터)  (GPT/Claude)
```

### Lab 8-1. 문서 임베딩 & 저장 (Ingest)

```python
from sentence_transformers import SentenceTransformer

model     = SentenceTransformer("all-MiniLM-L6-v2")   # dim=384
EMBED_DIM = 384
RAG_COL   = "rag_kb"

def create_rag_collection():
    connect_milvus()
    if utility.has_collection(RAG_COL):
        return Collection(RAG_COL)

    fields = [
        FieldSchema(name="id",     dtype=DataType.INT64,       is_primary=True, auto_id=True),
        FieldSchema(name="source", dtype=DataType.VARCHAR,      max_length=512),
        FieldSchema(name="chunk",  dtype=DataType.INT64),
        FieldSchema(name="text",   dtype=DataType.VARCHAR,      max_length=8192),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ]
    col = Collection(RAG_COL, CollectionSchema(fields))
    col.create_index("vector", {"index_type": "HNSW", "metric_type": "COSINE",
                                 "params": {"M": 16, "efConstruction": 200}})
    col.load()
    return col

rag_col = create_rag_collection()

# 문서 청크 & 임베딩
documents = [
    {"source": "doc1.pdf", "chunk": 0, "text": "Milvus는 오픈소스 벡터 DB입니다."},
    {"source": "doc1.pdf", "chunk": 1, "text": "HNSW는 그래프 기반 ANN 인덱스입니다."},
    {"source": "doc2.pdf", "chunk": 0, "text": "RAG는 검색 증강 생성(Retrieval-Augmented Generation)입니다."},
    {"source": "doc2.pdf", "chunk": 1, "text": "LLM과 벡터 DB를 결합하면 환각을 줄일 수 있습니다."},
]

texts   = [d["text"]   for d in documents]
sources = [d["source"] for d in documents]
chunks  = [d["chunk"]  for d in documents]
vectors = model.encode(texts, batch_size=32, normalize_embeddings=True).tolist()

rag_col.insert([sources, chunks, texts, vectors])
rag_col.flush()
print(f"저장: {rag_col.num_entities}개 청크")
```

### Lab 8-2. 쿼리 & 컨텍스트 검색 (Retrieval)

```python
def retrieve(query: str, top_k: int = 3, source_filter: str = None):
    q_vec = model.encode([query], normalize_embeddings=True).tolist()

    search_kwargs = dict(
        data          = q_vec,
        anns_field    = "vector",
        param         = {"metric_type": "COSINE", "params": {"ef": 64}},
        limit         = top_k,
        output_fields = ["text", "source", "chunk"],
    )
    if source_filter:
        search_kwargs["expr"] = f'source == "{source_filter}"'

    hits = rag_col.search(**search_kwargs)
    return [
        {"text":   hit.entity.get("text"),
         "source": hit.entity.get("source"),
         "score":  round(hit.distance, 4)}
        for hit in hits[0]
    ]

# 검색 실행
contexts = retrieve("벡터 데이터베이스란 무엇인가?")
for c in contexts:
    print(f"[{c['score']}] ({c['source']}) {c['text']}")
```

### Lab 8-3. LLM 답변 생성 (Generation)

```python
from openai import OpenAI

def rag_answer(question: str, top_k: int = 3) -> str:
    contexts     = retrieve(question, top_k=top_k)
    context_text = "\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)
    )
    prompt = f"""다음 컨텍스트를 참고하여 질문에 답하세요.
Context:
{context_text}

Question: {question}
Answer:"""

    llm  = OpenAI()   # OPENAI_API_KEY 환경변수 필요
    resp = llm.chat.completions.create(
        model    = "gpt-4o-mini",
        messages = [{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

answer = rag_answer("Milvus에서 가장 빠른 인덱스는?")
print(answer)
```

### Lab 8-4. 전체 파이프라인 점검 & 정리

```python
def inspect(col_name: str):
    connect_milvus()
    col = Collection(col_name)
    print(f"Collection : {col.name}")
    print(f"Entity 수  : {col.num_entities}개")
    print(f"Fields     : {[f.name for f in col.schema.fields]}")
    print(f"Indexes    : {[idx.field_name for idx in col.indexes]}")

inspect(RAG_COL)

# 정리
for name in [COLLECTION_NAME, "docs_sparse", "hybrid_docs", RAG_COL]:
    if utility.has_collection(name):
        utility.drop_collection(name)

connections.disconnect("default")
print("실습 완료 및 정리 완료")
```

> **PROD TIP** 프로덕션 RAG 구성:  
> (1) 청크 크기 512~1024 토큰, (2) 오버랩 10~20%,  
> (3) HNSW 인덱스, (4) 하이브리드 검색으로 recall 향상,  
> (5) Reranker(CrossEncoder)로 최종 정렬

---

## API 대응표 — ORM vs MilvusClient

| 작업 | ORM (구버전) | MilvusClient (신버전) |
|------|------------|----------------------|
| 연결 | `connections.connect()` | `MilvusClient(uri=...)` |
| 컬렉션 존재 확인 | `utility.has_collection()` | `client.has_collection()` |
| 컬렉션 목록 | `utility.list_collections()` | `client.list_collections()` |
| 컬렉션 객체 | `Collection("name")` | 없음 (client가 직접 처리) |
| 데이터 삽입 | `col.insert([[필드순서...]])` | `client.insert("name", [{dict}])` |
| 삭제 | `col.delete(expr=...)` | `client.delete("name", filter=...)` |
| 검색 | `col.search(anns_field=..., expr=...)` | `client.search("name", filter=...)` |
| 조회 | `col.query(expr=...)` | `client.query("name", filter=...)` |
| 하이브리드 | `col.hybrid_search(rerank=...)` | `client.hybrid_search(ranker=...)` |
| 연결 해제 | `connections.disconnect()` | `client.close()` |

---

*milvus.io/docs  |  github.com/milvus-io/milvus*
