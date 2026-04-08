# Milvus 벡터 데이터베이스 기초 실습

> 핵심 개념 · 실습 코드 · 아키텍처 가이드

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

### 설치 환경 확인 (Docker Standalone)

먼저 Milvus 컨테이너가 정상 실행 중인지 확인합니다.

```bash
# 컨테이너 상태 확인
docker compose ps

# 예상 출력
# NAME                IMAGE                STATUS
# milvus-standalone   milvus/milvus:...    Up (healthy)
# milvus-etcd         quay.io/coreos/etcd  Up (healthy)
# milvus-minio        minio/minio          Up (healthy)

# 내려가 있다면 다시 시작
docker compose up -d

# 로그 확인
docker compose logs milvus-standalone --tail=20
```

> **포트** gRPC `19530` · REST `9091` · MinIO `9000/9001`

```bash
# 포트 리슨 확인
lsof -i :19530           # macOS / Linux
netstat -an | grep 19530  # Windows PowerShell
```

---

### pymilvus 설치

```bash
pip install pymilvus numpy sentence-transformers

# 버전 확인
python -c "import pymilvus; print(pymilvus.__version__)"
```

---

### 로컬 서버 연결 (공통 import)

```python
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker, WeightedRanker
import numpy as np

# ── 로컬 Docker Standalone 연결 ──────────────────────────
client = MilvusClient(
    uri   = "http://localhost:19530",
    token = "root:Milvus",   # 기본 계정 (변경했다면 수정)
)

# 연결 확인
print(client.get_server_version())   # e.g. v2.4.x

DIM = 128   # 실습용 벡터 차원
```

> **NOTE** `token`은 `"사용자명:비밀번호"` 형식입니다. 계정을 변경하지 않았다면 기본값 `root:Milvus`를 그대로 사용하세요.

---

### 연결 트러블슈팅

```bash
# 1) Connection refused → 컨테이너 재시작
docker compose up -d

# 2) Authentication failed → 토큰 확인
#    client = MilvusClient(uri="http://localhost:19530", token="root:변경한비밀번호")

# 3) 헬스체크
curl http://localhost:9091/healthz
# {"status": "healthy"} 이면 정상
```

---

## 01. 컬렉션 관리 (Collection Management)

### 핵심 개념

Collection은 RDB의 Table에 해당합니다. Schema로 필드 구조를 정의하고, 인덱스 파라미터를 함께 지정하여 생성합니다.

| 개념 | 설명 | RDB 대응 |
|------|------|---------|
| Collection | 데이터 저장의 최상위 단위, 스키마+인덱스 포함 | Table |
| Field | 개별 데이터 속성. 벡터/스칼라 구분 | Column |
| Entity | 하나의 데이터 레코드 | Row |
| Segment | 내부 데이터 관리 단위 (Growing/Sealed) | Partition(내부) |
| Partition | 컬렉션 내 논리 분할 (tag 기반) | Table Partition |

### Lab 1-1. 컬렉션 생성

```python
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,  # 스키마 미정의 필드도 저장 가능
)

# 필드 추가
schema.add_field("id",        DataType.INT64,        is_primary=True)
schema.add_field("title",     DataType.VARCHAR,       max_length=256)
schema.add_field("category",  DataType.VARCHAR,       max_length=64)
schema.add_field("embedding", DataType.FLOAT_VECTOR,  dim=DIM)

# 인덱스 파라미터
idx = client.prepare_index_params()
idx.add_index("embedding", index_type="IVF_FLAT", metric_type="COSINE",
              params={"nlist": 64})

# 컬렉션 생성
client.create_collection("docs", schema=schema, index_params=idx)
print(client.list_collections())   # ["docs"]
```

### Lab 1-2. 파티션 관리

```python
# 파티션 생성 (카테고리별 분리)
client.create_partition("docs", partition_name="tech")
client.create_partition("docs", partition_name="news")

# 파티션 목록 확인
print(client.list_partitions("docs"))   # ["_default", "tech", "news"]

# 특정 파티션에만 검색 (성능 향상)
client.search("docs", [q_vec], limit=5, partition_names=["tech"])

# 파티션 삭제
client.drop_partition("docs", "news")
```

### Lab 1-3. 컬렉션 수명주기

```python
client.describe_collection("docs")      # 스키마·인덱스 상세 확인
client.get_collection_stats("docs")     # 엔티티 수 등 통계

client.release_collection("docs")       # 메모리 해제 (검색 불가 상태)
client.load_collection("docs")          # 메모리 로드 (검색 가능 상태)

client.rename_collection("docs", "documents")
client.drop_collection("documents")     # 완전 삭제
```

> **NOTE** `create_collection()`에 schema와 index_params를 함께 전달하면 생성+인덱스를 한 번에 처리합니다. 별도로 `create_index()`를 호출하면 기존 컬렉션에 인덱스를 추가할 수 있습니다.

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
data = [
    {
        "id":        i,
        "title":     f"문서 제목 {i}",
        "category":  "tech" if i % 2 == 0 else "news",
        "embedding": np.random.rand(DIM).tolist(),
        "score":     float(np.random.randint(60, 100)),  # dynamic field
    }
    for i in range(1000)
]

res = client.insert("docs", data)
print(f"삽입: {res['insert_count']}개")

# 단건 Upsert (있으면 update, 없으면 insert)
client.upsert("docs", {"id": 0, "title": "수정된 제목", "embedding": np.random.rand(DIM).tolist()})
```

### Lab 2-2. Sparse 벡터 삽입

```python
# Sparse 컬렉션 스키마
schema_sp = MilvusClient.create_schema(auto_id=False)
schema_sp.add_field("id",     DataType.INT64,               is_primary=True)
schema_sp.add_field("text",   DataType.VARCHAR,              max_length=512)
schema_sp.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

idx_sp = client.prepare_index_params()
idx_sp.add_index("sparse", index_type="SPARSE_INVERTED_INDEX",
                 metric_type="IP", params={"drop_ratio_build": 0.2})

client.create_collection("docs_sparse", schema=schema_sp, index_params=idx_sp)

# Sparse 벡터: dict {term_id: weight} 형태
sparse_data = [
    {"id": i, "text": f"doc {i}",
     "sparse": {j: float(np.random.rand()) for j in
                np.random.choice(30000, 20, replace=False)}}
    for i in range(500)
]
client.insert("docs_sparse", sparse_data)
```

### Lab 2-3. 데이터 수정·삭제

```python
# ID 기반 삭제
client.delete("docs", ids=[0, 1, 2])

# 조건 기반 삭제 (표현식 사용)
client.delete("docs", filter='category == "news" and score < 70')

# ID로 단건 조회 후 수정
entity = client.get("docs", ids=[10])
entity[0]["title"] = "업데이트된 제목"
client.upsert("docs", entity)
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
rows = client.query(
    "docs",
    filter='category == "tech" and score >= 80',
    output_fields=["id", "title", "score"],
    limit=20,
)
for r in rows:
    print(r["id"], r["title"], r["score"])
```

### Lab 3-2. ID 직접 조회 & 전체 스캔

```python
# ID 리스트로 직접 조회
entities = client.get("docs", ids=[10, 20, 30])

# 전체 데이터 스캔 (페이지네이션)
offset = 0
while True:
    batch = client.query("docs", filter="id >= 0",
                         output_fields=["id", "title"],
                         limit=100, offset=offset)
    if not batch: break
    print(f"  offset={offset}: {len(batch)}건")
    offset += 100
```

> **TIP** `query()`는 벡터 검색 없이 스칼라만 조회하므로 인덱스 없이도 동작합니다. 단, 대용량에서는 filter에 인덱스가 걸린 필드를 사용하는 것을 권장합니다.

---

## 04. 벡터 검색 (ANN Search)

### 핵심 개념

```
Query Text → Embedding Model → Query Vector → ANN Search → Ranking & Filter → Top-K Results → LLM(RAG)
   입력         SBERT/OpenAI      부동소수 배열    IVF/HNSW      nprobe/ef       distance+fields    답변 생성
```

ANN(Approximate Nearest Neighbor) 검색은 쿼리 벡터와 가장 유사한 K개의 벡터를 근사적으로 찾습니다. 100% 정확도를 포기하는 대신 수십억 건에서도 밀리초 응답을 실현합니다.

| 메트릭 | 수식 개요 | 적합한 데이터 |
|--------|---------|-------------|
| COSINE | cos(θ) = A·B / (\|A\|\|B\|) | 텍스트, 문서 임베딩 (방향성) |
| L2 | √Σ(aᵢ-bᵢ)² | 이미지, 수치 특징 (절대 거리) |
| IP | Σ aᵢ·bᵢ | 정규화 벡터에서 COSINE과 동일 |
| JACCARD | 교집합/합집합 | Binary 벡터, 집합 유사도 |

### Lab 4-1. 기본 ANN 검색

```python
q_vec = np.random.rand(DIM).tolist()   # 실제 사용 시: 임베딩 모델 출력

results = client.search(
    collection_name = "docs",
    data            = [q_vec],          # 배치 검색 가능: [vec1, vec2, ...]
    limit           = 5,                # Top-K
    output_fields   = ["title", "category", "score"],
    search_params   = {
        "metric_type": "COSINE",
        "params":      {"nprobe": 16},  # IVF: 탐색 클러스터 수 (↑정확도, ↑속도)
    }
)

for hit in results[0]:
    print(f"id={hit['id']} dist={hit['distance']:.4f} title={hit['entity']['title']}")
```

### Lab 4-2. Range Search (거리 범위 제한)

```python
# COSINE 유사도 0.8 이상인 결과만 반환
results = client.search(
    "docs", [q_vec], limit=20,
    search_params={
        "metric_type": "COSINE",
        "params": {
            "nprobe":       16,
            "radius":       0.8,   # 최소 유사도 (COSINE: 클수록 유사)
            "range_filter": 1.0,   # 최대 유사도
        }
    }
)
```

### Lab 4-3. Iterator 기반 대량 검색

```python
# 수백만 건 결과를 메모리 초과 없이 처리
from pymilvus import SearchIterator

iterator = client.search_iterator(
    "docs", [q_vec],
    batch_size=100,
    output_fields=["title"],
    search_params={"metric_type": "COSINE", "params": {"nprobe": 8}},
)
total = 0
while True:
    batch = iterator.next()
    if not batch: break
    total += len(batch)
iterator.close()
print(f"총 {total}건 처리")
```

---

## 05. 인덱스 (Index)

### 핵심 개념

인덱스는 벡터 검색 속도와 정확도의 트레이드오프를 결정합니다. 데이터 규모·메모리·QPS 요건에 따라 선택합니다.

| 인덱스 | 알고리즘 | 주요 파라미터 | 권장 규모 |
|--------|---------|-------------|---------|
| FLAT | 완전 탐색 | 없음 | < 100만 |
| IVF_FLAT | 역인덱스 클러스터 | nlist(64~2048), nprobe(검색 시) | 100만~1억 |
| IVF_SQ8 | IVF+스칼라 양자화 | nlist | 메모리 제약 |
| IVF_PQ | IVF+곱 양자화 | nlist, m(서브벡터), nbits | 고차원 대용량 |
| HNSW | 계층 그래프 | M(16~64), efConstruction(100~500) | 온라인 서비스 |
| DISKANN | 그래프+디스크 | search_list, build_ratio | > 1억 (디스크) |
| SCANN | Google ANN | nlist, with_raw_data | 추천 시스템 |

```
검색속도  : FLAT(1) < IVF_FLAT(4) < HNSW(8) < SCANN(9)
메모리효율: FLAT(2) < HNSW(4) < IVF_FLAT(5) < DISKANN(9)
정확도    : HNSW(8) ≈ SCANN(8) > IVF_FLAT(7) > IVF_SQ8(6)
```

### Lab 5-1. 인덱스 교체 (IVF_FLAT → HNSW)

```python
# 기존 인덱스 해제 후 삭제
client.release_collection("docs")
client.drop_index("docs", "embedding")

# HNSW 인덱스 생성
idx_hnsw = client.prepare_index_params()
idx_hnsw.add_index(
    "embedding",
    index_type  = "HNSW",
    metric_type = "COSINE",
    params      = {
        "M":              16,   # 그래프 연결 수 (↑정확도, ↑메모리)
        "efConstruction": 200,  # 빌드 시 탐색 폭 (↑정확도, ↑빌드시간)
    }
)
client.create_index("docs", idx_hnsw)
client.load_collection("docs")

# HNSW 검색 시 ef 파라미터 조정
results = client.search(
    "docs", [q_vec], limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    # ef >= limit 권장. ↑ef = ↑정확도, ↑레이턴시
)
```

### Lab 5-2. 스칼라 필드 인덱스

```python
# 스칼라 필드에 인덱스 추가 (필터 성능 향상)
client.create_index("docs", index_params=client.prepare_index_params().add_index(
    "category",          # VARCHAR 필드
    index_type="Trie",   # 문자열: Trie / 숫자: STL_SORT / INVERTED
))

# JSON 필드 인덱스
client.create_index("docs", index_params=client.prepare_index_params().add_index(
    "meta",
    index_type     = "INVERTED",
    json_path      = "meta['lang']",   # JSON 내 특정 키
    json_cast_type = DataType.VARCHAR,
))
```

> **TIP** HNSW는 빠른 검색과 높은 정확도로 온라인 서비스에 최적입니다. 데이터가 수십억 건이고 메모리가 부족하다면 DISKANN을 선택하세요.

---

## 06. 랭킹 & 필터 (Ranking & Filtering)

### 핵심 개념

```
전체 Entity(100만) → Pre-Filter 적용(2만) → ANN 검색 nprobe=16(100개) → Re-Rank & Limit(Top-5) → Output Fields 반환
                    filter='cat=="news"'        search()                   RRF / score 정렬        output_fields=[...]
```

필터는 검색 전(Pre-filter, 기본) 또는 검색 후(Post-filter) 적용할 수 있습니다.

| 방식 | 설명 | 장단점 |
|------|------|--------|
| Pre-filter (기본) | 필터 후 ANN 검색 | 결과 수 보장 어려움, 정밀 필터에 최적 |
| Post-filter | ANN 후 필터 적용 | 결과 수 예측 가능, 대용량 필터에 유리 |
| Range Search | 거리 범위로 후보 제한 | 유사도 임계값 적용 가능 |

### Lab 6-1. Pre-filter 검색

```python
# filter 표현식 + 벡터 검색
results = client.search(
    "docs", [q_vec], limit=10,
    filter='category == "tech" and score >= 80',
    output_fields=["title", "category", "score"],
    search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
)

# IN 연산자로 다중 값 필터
results2 = client.search(
    "docs", [q_vec], limit=5,
    filter='id in [10, 20, 30, 40, 50]',
    search_params={"metric_type": "COSINE", "params": {"nprobe": 8}},
)
```

### Lab 6-2. 거리 메트릭 비교 실습

```python
# 동일 쿼리, 다른 메트릭으로 결과 비교
for metric in ["COSINE", "L2", "IP"]:
    res = client.search(
        "docs", [q_vec], limit=3,
        search_params={"metric_type": metric, "params": {"nprobe": 8}},
        output_fields=["title"],
    )
    ids   = [h["id"] for h in res[0]]
    dists = [round(h["distance"], 4) for h in res[0]]
    print(f"{metric:8s} → ids={ids}  dists={dists}")

# COSINE: 높을수록 유사 (1.0 = 완벽 일치)
# L2:     낮을수록 유사 (0.0 = 완벽 일치)
# IP:     높을수록 유사 (정규화된 벡터에서)
```

### Lab 6-3. 그룹화 검색 (GroupBy)

```python
# 카테고리별로 대표 1개씩 반환 (다양성 확보)
results = client.search(
    "docs", [q_vec], limit=6,
    group_by_field="category",        # 각 category에서 best 1개
    group_size=1,
    output_fields=["title", "category"],
    search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
)
for hit in results[0]:
    print(hit["entity"]["category"], hit["entity"]["title"])
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

하이브리드 검색은 Dense(시맨틱) + Sparse(키워드) 결과를 Re-Ranker로 통합합니다. 단일 방식보다 검색 품질이 크게 향상됩니다.

| Re-Ranker | 수식 | 특징 |
|-----------|------|------|
| RRFRanker | score = Σ 1/(k + rankᵢ) | 순위 기반. 점수 스케일 무관, 안정적 |
| WeightedRanker | score = Σ wᵢ × normScoreᵢ | 가중치 직접 지정. 도메인 지식 반영 |

### Lab 7-1. 하이브리드 컬렉션 구성

```python
# Dense + Sparse 필드를 모두 가진 컬렉션
schema_h = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema_h.add_field("id",     DataType.INT64,               is_primary=True)
schema_h.add_field("text",   DataType.VARCHAR,              max_length=512)
schema_h.add_field("dense",  DataType.FLOAT_VECTOR,         dim=DIM)
schema_h.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

idx_h = client.prepare_index_params()
idx_h.add_index("dense",  index_type="HNSW",
                metric_type="COSINE", params={"M": 16, "efConstruction": 200})
idx_h.add_index("sparse", index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",     params={"drop_ratio_build": 0.2})

client.create_collection("hybrid_docs", schema=schema_h, index_params=idx_h)

# 데이터 삽입
hybrid_data = [
    {"id": i, "text": f"doc {i}",
     "dense":  np.random.rand(DIM).tolist(),
     "sparse": {j: float(np.random.rand())
                for j in np.random.choice(30000, 15, replace=False)}}
    for i in range(500)
]
client.insert("hybrid_docs", hybrid_data)
```

### Lab 7-2. RRFRanker 하이브리드 검색

```python
dense_req = AnnSearchRequest(
    data        = [np.random.rand(DIM).tolist()],
    anns_field  = "dense",
    param       = {"metric_type": "COSINE", "params": {"ef": 100}},
    limit       = 10,
)
sparse_req = AnnSearchRequest(
    data        = [{j: float(np.random.rand()) for j in
                    np.random.choice(30000, 10, replace=False)}],
    anns_field  = "sparse",
    param       = {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
    limit       = 10,
)

# RRF: k=60 (기본값). 낮을수록 상위 랭크에 가중치 집중
results = client.hybrid_search(
    "hybrid_docs",
    reqs   = [dense_req, sparse_req],
    ranker = RRFRanker(k=60),
    limit  = 5,
    output_fields = ["text"],
)
for hit in results[0]:
    print(f"id={hit['id']}  rrf_score={hit['distance']:.4f}  text={hit['entity']['text']}")
```

### Lab 7-3. WeightedRanker (가중치 조정)

```python
# dense 70%, sparse 30% 가중치
results_w = client.hybrid_search(
    "hybrid_docs",
    reqs   = [dense_req, sparse_req],
    ranker = WeightedRanker(0.7, 0.3),  # 순서 = reqs 순서와 일치
    limit  = 5,
)

# 실험: 가중치를 변경하며 결과 비교
for w in [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)]:
    r = client.hybrid_search("hybrid_docs", [dense_req, sparse_req],
                              ranker=WeightedRanker(*w), limit=3)
    ids = [h["id"] for h in r[0]]
    print(f"dense={w[0]} sparse={w[1]} → {ids}")
```

> **TIPS** 키워드 정확성이 중요한 법률·의학 도메인은 sparse 가중치를, 의미적 유사성이 중요한 추천·QA는 dense 가중치를 높이세요.

---

## 08. 실전 구성 — RAG 파이프라인

### 아키텍처

```
[ Ingest Pipeline ]
문서 원본 → Chunking → Embedding Model → Milvus Insert
(PDF/HTML)  (Splitter)  (SBERT/OpenAI)   (collection.insert())
                                                  │
                                             저장된 벡터
                                                  │
[ Query Pipeline ]                                ▼
User Query → Embedding → Hybrid Search → Re-Rank & Filter → LLM → Answer
  (질문)      (벡터화)    (Dense+Sparse)   (RRF/메타필터)  (GPT/Claude)
                                  ↑
                             context 전달
```

### Lab 8-1. 문서 임베딩 & 저장 (Ingest)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # dim=384
EMBED_DIM = 384

# RAG 컬렉션 생성
schema_rag = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
schema_rag.add_field("id",      DataType.INT64,       is_primary=True)
schema_rag.add_field("vector",  DataType.FLOAT_VECTOR, dim=EMBED_DIM)

idx_rag = client.prepare_index_params()
idx_rag.add_index("vector", index_type="HNSW", metric_type="COSINE",
                  params={"M": 16, "efConstruction": 200})

client.create_collection("rag_kb", schema=schema_rag, index_params=idx_rag)

# 문서 청크 & 임베딩
documents = [
    {"source": "doc1.pdf", "chunk": 0, "text": "Milvus는 오픈소스 벡터 DB입니다."},
    {"source": "doc1.pdf", "chunk": 1, "text": "HNSW는 그래프 기반 ANN 인덱스입니다."},
    {"source": "doc2.pdf", "chunk": 0, "text": "RAG는 검색 증강 생성(Retrieval-Augmented Generation)입니다."},
    {"source": "doc2.pdf", "chunk": 1, "text": "LLM과 벡터 DB를 결합하면 환각을 줄일 수 있습니다."},
]

texts   = [d["text"] for d in documents]
vectors = model.encode(texts, batch_size=32, normalize_embeddings=True).tolist()

ingest_data = [{"vector": v, **d} for v, d in zip(vectors, documents)]
res = client.insert("rag_kb", ingest_data)
print(f"저장: {res['insert_count']}개 청크")
```

### Lab 8-2. 쿼리 & 컨텍스트 검색 (Retrieval)

```python
def retrieve(query: str, top_k: int = 3, source_filter: str = None):
    q_vec = model.encode([query], normalize_embeddings=True).tolist()

    search_kwargs = dict(
        collection_name = "rag_kb",
        data            = q_vec,
        limit           = top_k,
        output_fields   = ["text", "source", "chunk"],
        search_params   = {"metric_type": "COSINE", "params": {"ef": 64}},
    )
    if source_filter:
        search_kwargs["filter"] = f'source == "{source_filter}"'

    hits = client.search(**search_kwargs)
    return [
        {"text":   h["entity"]["text"],
         "source": h["entity"]["source"],
         "score":  round(h["distance"], 4)}
        for h in hits[0]
    ]

# 검색 실행
contexts = retrieve("벡터 데이터베이스란 무엇인가?")
for c in contexts:
    print(f"[{c['score']}] ({c['source']}) {c['text']}")
```

### Lab 8-3. LLM 답변 생성 (Generation)

```python
# pip install openai
from openai import OpenAI

def rag_answer(question: str, top_k: int = 3) -> str:
    # 1) 관련 컨텍스트 검색
    contexts = retrieve(question, top_k=top_k)
    context_text = "\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)
    )

    # 2) 프롬프트 구성
    prompt = f"""다음 컨텍스트를 참고하여 질문에 답하세요.
Context:
{context_text}

Question: {question}
Answer:"""

    # 3) LLM 호출
    llm = OpenAI()   # OPENAI_API_KEY 환경변수 필요
    resp = llm.chat.completions.create(
        model    = "gpt-4o-mini",
        messages = [{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

# 실행
answer = rag_answer("Milvus에서 가장 빠른 인덱스는?")
print(answer)
```

### Lab 8-4. 전체 파이프라인 점검

```python
# 컬렉션 상태 점검 함수
def inspect(col: str):
    info  = client.describe_collection(col)
    stats = client.get_collection_stats(col)
    print(f"Collection : {info['collection_name']}")
    print(f"Entity 수  : {stats['row_count']}개")
    print(f"Fields     : {[f['name'] for f in info['schema']['fields']]}")
    print(f"Indexes    : {[idx['field_name'] for idx in info['indexes']]}")

inspect("rag_kb")

# 정리
client.drop_collection("docs")
client.drop_collection("docs_sparse")
client.drop_collection("hybrid_docs")
client.drop_collection("rag_kb")
import os; os.remove("milvus_lab.db")
print("실습 완료 및 정리 완료")
```

> **PROD TIP** 프로덕션 RAG 구성: (1) 청크 크기 512~1024 토큰, (2) 오버랩 10~20%, (3) HNSW 인덱스, (4) 하이브리드 검색으로 recall 향상, (5) Reranker(CrossEncoder)로 최종 정렬

---

*milvus.io/docs  |  github.com/milvus-io/milvus*
