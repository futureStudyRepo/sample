# Milvus 트러블슈팅 작업일지

> 작업일: 2025-04-09  
> 최종 버전: **Milvus v2.5.10** + Attu v2.5 + etcd v3.5.25

---

## 요약 흐름

```
Sparse Index 오류 발견
  → Milvus 버전 문제 확인 (v2.3 → v2.5 업그레이드)
    → Attu 로그인 안됨 (인증 추가됨)
      → 컨테이너 계속 죽음 (MQ 타입 오류 + etcd 데이터 충돌)
        → docker-compose 재구성 + user.yaml 적용
          → 정상 운영 ✅
```

---

## 1. 문제 시작 — Sparse Vector 인덱스 오류

**증상**

```
index type not match: expected=Trie, actual=SPARSE_INVERTED_INDEX
```

**원인**

| 항목 | 내용 |
|------|------|
| 실행 중 버전 | Milvus v2.3.0-dev |
| 문제 | `SPARSE_FLOAT_VECTOR` / `SPARSE_INVERTED_INDEX`는 **v2.4.0 이상** 전용 |
| 결론 | 버전 업그레이드 필요 |

**해결**  
→ Milvus **v2.5.10** 업그레이드 결정

---

## 2. 업그레이드 후 Attu 접속 불가

**원인**  
Milvus **2.4.x 이상**부터 기본 인증(Authorization)이 활성화됨

**해결 — 기본 계정으로 로그인**

| 항목 | 값 |
|------|----|
| ID | `root` |
| PW | `Milvus` |

---

## 3. Milvus 컨테이너가 계속 죽는 문제

### 원인 1 — Woodpecker MQ 타입 오류

```
panic: mq type woodpecker is invalid
```

`docker-compose.yml`에 `MQ_TYPE: woodpecker`가 설정되어 있었는데, Woodpecker는 **v2.6.x 전용**

```yaml
# 변경 전
MQ_TYPE: woodpecker

# 변경 후
MQ_TYPE: rocksmq
```

### 원인 2 — 이전 버전 etcd 데이터 충돌

v2.3.x etcd 데이터가 남아 있어 새 버전이 기동 중 크래시

```bash
cd ~
# 1. 기존 컨테이너 및 데이터 삭제 (처음 설치면 에러나도 무시)
docker-compose down -v
docker rm -f attu
rm -rf volumes/
docker-compose down -v
rm -rf volumes/
docker-compose up -d
```

> ⚠️ `volumes/` 삭제 시 기존 데이터가 모두 사라집니다. 처음 세팅 또는 실습 환경에서만 실행하세요.

---

## 4. 최종 docker-compose.yml 구성

| 서비스 | 이미지 | 포트 |
|--------|--------|------|
| milvus-etcd | `etcd:v3.5.25` | 2379 |
| milvus-minio | `minio` | 9000, 9001 |
| milvus-standalone | `milvus:v2.5.10` | 19530, 9091 |
| attu | `attu:v2.5` | 8000 |

**핵심 설정 — MQ 타입**

```yaml
environment:
  MQ_TYPE: rocksmq   # ← woodpecker 아님!
```

---

## 5. 인증 비활성화 (Attu 접속 편의)

### user.yaml 작성

```yaml
# user.yaml
common:
  security:
    authorizationEnabled: false
```

### docker-compose.yml 마운트 추가

```yaml
services:
  standalone:
    volumes:
      - ./user.yaml:/milvus/configs/user.yaml
```

---

## 6. 친구들 배포 방법

```bash
# 1. 기존 컨테이너 정리 (처음이면 에러 무시)
docker-compose down -v
rm -rf volumes/

# 2. 파일 다운로드
curl -O https://raw.githubusercontent.com/futureStudyRepo/sample/refs/heads/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/futureStudyRepo/sample/refs/heads/main/user.yaml

# 3. 실행
docker-compose up -d
```

---

## 7. Attu 접속 방법

```bash
# WSL IP 확인
ip addr show eth0 | grep inet
```

| 항목 | 값 |
|------|----|
| Attu URL | `http://WSL_IP:8000` |
| Milvus Address | `milvus-standalone:19530` |

> ⚠️ **`Check Health` 체크 해제** 후 Connect!

---

## 8. Python 연결

```python
from pymilvus import connections

connections.connect(host="localhost", port="19530")
```

---

## 버전별 기능 호환 정리

| 기능 | 최소 버전 |
|------|---------|
| `SPARSE_FLOAT_VECTOR` | v2.4.0+ |
| `SPARSE_INVERTED_INDEX` | v2.4.0+ |
| 기본 인증 활성화 | v2.4.0+ |
| `MQ_TYPE: woodpecker` | v2.6.0+ |
| `MQ_TYPE: rocksmq` | v2.x (범용) |
