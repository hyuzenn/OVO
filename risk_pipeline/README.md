# risk_pipeline MVP (Phase D)

이 디렉토리는 기존 OVO risk-aware 경로와 분리된 **독립 MVP 파이프라인**입니다.

## 구성

- `core/runner.py`
  - Phase D 실행 순서를 고정한 엔트리 (`RiskPipelineRunner`)
  - 순서:
    1. scene graph 로드 (스크립트에서 수행)
    2. base node representation `z_i` 생성
    3. graph encode `r_i_rel`
    4. retrieval `r_i_retr`
    5. modulation `z_i'`
    6. mapping integrate
- `scripts/build_failure_memory.py`
  - MVP 실험용 prototype memory JSON 생성
- `scripts/run_pipeline.py`
  - CLI로 end-to-end 실행

## 설치

아래는 최소 의존성 예시입니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pytest
```

## 실행

### 1) prototype memory 생성

```bash
python -m risk_pipeline.scripts.build_failure_memory \
  --dim 32 \
  --num-prototypes 4 \
  --output /tmp/failure_memory.json
```

### 2) 파이프라인 실행

입력 JSON 형식은 `SGFrontLoader`가 읽는 SG-FRONT 유사 포맷입니다.

```bash
python -m risk_pipeline.scripts.run_pipeline \
  --relationships-json /path/to/relationships.json \
  --obj-boxes-json /path/to/obj_boxes.json \
  --memory-json /tmp/failure_memory.json \
  --hidden-dim 32 \
  --top-k 3 \
  --voxel-size 0.5 \
  --output /tmp/risk_pipeline_output.json
```

## 테스트

```bash
pytest risk_pipeline/tests/test_runner.py -q
```

필요시 기존 테스트도 함께 실행:

```bash
pytest risk_pipeline/tests -q
```

## 현재 구현된 것

- 독립 실행 경로(Loader → Base Rep → Graph Encode → Retrieval → Modulation → Mapping) 완성
- CLI 기반 실행 스크립트 제공
- prototype memory 생성 스크립트 제공
- runner 단위의 end-to-end 테스트 제공

## 아직 안 한 것

- 학습/튜닝 파이프라인
- 대규모 데이터셋 학습 루프 및 체크포인트 관리
- 실제 로봇 제어/경로계획과의 통합
- 성능 벤치마크 자동화
- 시각화 대시보드

문서에는 현재 코드로 실제 실행 가능한 범위만 기술했습니다.
