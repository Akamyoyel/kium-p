# kium-p 사용설명서

> 금융 자산 수익률의 동적 위험 예측을 위한 다분위수 확률 시계열 예측
> 2026학년도 1학기 더 KIUM학기제 | 김도희 (20201889, 스마트보안학과)

---

## 1. 환경 구축

### 1-1. Python 가상환경 생성

```bash
conda create -n kium_quant python=3.10
conda activate kium_quant
```

### 1-2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 1-3. PyTorch 설치 확인

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

### 1-4. 환경변수 설정

```bash
copy .env.example .env
```

`.env` 파일에 `KIS_APP_KEY`, `KIS_APP_SECRET`을 입력합니다.

---

## 2. 데이터 준비 - CSV 파일 배치

> KIS API는 과거 분봉의 날짜지정 수집에 제약이 있으므로, 핵심 학습 데이터는 CSV 파일로 준비합니다.

### 2-1. 배치 위치

```text
kium-p/
└── data/
    └── raw/
        ├── kospi_top10_2025.csv
        ├── 005930.csv
        ├── 000660.csv
        └── ...
```

### 2-2. 파일명 규칙

- 종목별 파일은 `{종목코드}.csv` 형식을 권장합니다.
- 통합 파일은 `kospi_top10_2025.csv`를 사용합니다.

### 2-3. 지원 컬럼 형식

| 형식 | datetime | OHLCV |
|---|---|---|
| KIS 응답 | `stck_bsop_date` + `stck_cntg_hour` | `stck_oprc`, `stck_hgpr`, `stck_lwpr`, `stck_prpr`, `cntg_vol` |
| 일반 형식 | `Date` + `Time` | `Open`, `High`, `Low`, `Close`, `Volume` |
| 소문자 형식 | `datetime` | `open`, `high`, `low`, `close`, `volume` |

### 2-4. CSV 컬럼 확인

```python
from src.data.collector import inspect_csv

inspect_csv("data/raw/kospi_top10_2025.csv")
```

매핑되지 않는 컬럼이 있으면 `src/data/collector.py`의 `_CSV_COLUMN_MAP`에 추가합니다.

---

## 3. KIS API 설정 - 당일 추가 수집

당일 5분봉 추가 수집은 가능하지만, 과거 날짜를 지정한 분봉 수집은 제한됩니다. 장 마감 후 한 번만 실행하는 구성이 가장 안정적입니다.

### 3-1. API 키 발급

1. 한국투자증권 Open API 앱 등록
2. `APP_KEY`, `APP_SECRET` 발급
3. `.env`에 입력

```env
KIS_APP_KEY=...
KIS_APP_SECRET=...
KIS_BASE_URL=https://openapi.koreainvestment.com:9443
```

### 3-2. 토큰 확인

```python
from config.kis_auth import get_access_token

token = get_access_token()
print(token[:20])
```

---

## 4. 2차시 - 데이터 수집 및 검증

### 4-1. 단일 종목 로드

```python
from src.data.collector import load_from_csv

df = load_from_csv("005930")
print(df.shape)
```

### 4-2. CSV + 당일 데이터 결합

```python
from src.data.collector import load_and_update

df = load_and_update("005930", collect_today_flag=True)
```

`collect_today_flag=False`로 두면 CSV만 읽습니다.

### 4-3. 전체 종목 일괄 로드

```python
from src.data.collector import load_all_tickers

dfs = load_all_tickers(collect_today_flag=False)
```

### 4-4. 데이터 검증

```python
from src.data.validator import validate, create_sandbox

report = validate(df, ticker="005930")
sandbox = create_sandbox("005930", months=3)
```

---

## 5. 3차시 - 탐색적 데이터 분석

```bash
python notebooks/02_eda.py
```

EDA 결과는 `reports/figures/`에 저장됩니다.

---

## 6. 4차시 - 특성공학 파이프라인

```bash
python notebooks/03_feature_engineering.py
```

또는 Python에서 직접 실행합니다.

```python
import runpy

ns = runpy.run_path("notebooks/03_feature_engineering.py")
result = ns["run_feature_pipeline"]("005930")
X_train = result["X_train"]
y_train = result["y_train"]
```

생성 산출물은 다음과 같습니다.

- `data/cleaned/{ticker}_clean.parquet`
- `data/features/minmax_scaler.pkl`
- `data/splits/windows.npz`

---

## 7. 5차시 - Baseline 학습

```bash
python notebooks/04_baseline_lstm.py
```

학습 결과는 다음 파일에 저장됩니다.

- `data/models/lstm_point_best.pt`
- `data/models/train_history.npy`
- `reports/model_spec.md`

---

## 8. API 제약 및 대응 전략

| 항목 | 내용 |
|---|---|
| 과거 분봉 날짜 지정 | 불가 |
| 당일 분봉 수집 | 가능 |
| 권장 수집 시점 | 장 마감 후 1회 |
| 핵심 학습 데이터 | `kospi_top10_2025.csv` |

---

## 9. 자주 발생하는 오류와 해결법

### CSV 파일을 찾을 수 없음

`data/raw/005930.csv` 또는 `data/raw/kospi_top10_2025.csv`가 있는지 확인합니다.

### API 토큰 오류

`.env`에 `KIS_APP_KEY`, `KIS_APP_SECRET`이 올바르게 들어 있는지 확인합니다.

### `windows.npz` 없음

`python notebooks/03_feature_engineering.py`를 먼저 실행합니다.

### `minmax_scaler.pkl` 없음

특성공학 단계가 끝나지 않았습니다. 4차시 파이프라인을 다시 실행합니다.

---

## 10. 디렉토리 구조

```text
kium-p/
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── features/
│   └── splits/
├── notebooks/
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   └── 04_baseline_lstm.py
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── losses/
│   └── models/
└── reports/
```

---

최종 수정: 2026-06-05 | kium-p v1.0-final# kium-p 사용설명서
> 금융 자산 수익률의 동적 위험 예측을 위한 다분위수 확률 시계열 예측  
> 2026학년도 1학기 더 KIUM학기제 | 김도희 (20201889, 스마트보안학과)

---

## 목차

1. [환경 구축](#1-환경-구축)
2. [데이터 준비 — CSV 파일 배치](#2-데이터-준비--csv-파일-배치)
3. [KIS API 설정 (당일 추가 수집)](#3-kis-api-설정-당일-추가-수집)
4. [2차시: 데이터 수집 및 검증 실행](#4-2차시-데이터-수집-및-검증-실행)
5. [3차시: 탐색적 데이터 분석 실행](#5-3차시-탐색적-데이터-분석-실행)
6. [4차시: 특성공학 파이프라인 실행](#6-4차시-특성공학-파이프라인-실행)
7. [API 제약 안내 및 대응 전략](#7-api-제약-안내-및-대응-전략)
8. [자주 발생하는 오류와 해결법](#8-자주-발생하는-오류와-해결법)
9. [디렉토리 구조](#9-디렉토리-구조)

---

## 1. 환경 구축

### 1-1. Python 가상환경 생성



```bash
conda create -n D:\ksi_quant python=3.10
conda activate D:\mini (32비트, 데이터 수집용)
conda activate D:\ksi_quant(64비트, 데이터 검증용)
```

### 1-2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 1-3. PyTorch (CUDA 12.1)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# GPU 확인
python -c "import torch; print(torch.cuda.is_available())"
# → True 출력되면 정상
```

### 1-4. 환경변수 설정 (KIS API)

```bash
cp .env.example .env
# .env 파일을 열어 KIS_APP_KEY, KIS_APP_SECRET 입력
```

---

## 2. 데이터 준비 — CSV 파일 배치

> ⚠ **KIS API 제약으로 과거 5분봉 날짜 지정 수집이 불가합니다.**  
> 기 보유한 CSV 파일을 아래 경로에 배치해야 합니다.

### 2-1. CSV 파일 배치 위치

```
kium-p/
└── data/
    └── raw/
        ├── 005930.csv     ← 삼성전자 5분봉 CSV
        ├── 000660.csv     ← SK하이닉스
        ├── 035420.csv     ← NAVER
        └── ...
```

**파일명 규칙:** `{종목코드}.csv` (6자리 종목코드)

### 2-2. 지원하는 CSV 헤더 형식

| 형식 | datetime 컬럼 | OHLCV 컬럼 |
|---|---|---|
| KIS API 응답 형식 | `stck_bsop_date` + `stck_cntg_hour` | `stck_oprc`, `stck_hgpr`, `stck_lwpr`, `stck_prpr`, `cntg_vol` |
| 일반 영문 형식 | `Date` + `Time` | `Open`, `High`, `Low`, `Close`, `Volume` |
| 소문자 형식 | `datetime` | `open`, `high`, `low`, `close`, `volume` |

### 2-3. CSV 컬럼 확인 방법

```python
from src.data.collector import inspect_csv

# 배치한 CSV 파일의 컬럼 구조 확인
inspect_csv("data/raw/005930.csv")
```

출력 예시:
```
──────────────────────────────────────────────────
파일: 005930.csv
컬럼: ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
첫 3행:
      Date    Time   Open   High    Low  Close  Volume
20250103  090000  74000  74300  73900  74200  123456
──────────────────────────────────────────────────
매핑 가이드:
  'Date'   → date
  'Time'   → time
  'Open'   → open
  ...
```

매핑되지 않는 컬럼이 있으면 `src/data/collector.py` 상단 `_CSV_COLUMN_MAP`에 추가합니다.

---

## 3. KIS API 설정 (당일 추가 수집)

> 과거 수집은 불가하지만, **당일 5분봉 추가**는 가능합니다.  
> 매 거래일 장 마감 후 실행하여 데이터를 점진적으로 축적합니다.

### 3-1. API KEY 발급

1. [한국투자증권 홈페이지](https://securities.koreainvestment.com) 로그인
2. 트레이딩 → Open API → 앱 등록
3. APP_KEY, APP_SECRET 발급 후 `.env`에 입력

```env
KIS_APP_KEY=ABCDEFGHabcdefgh...
KIS_APP_SECRET=xxxxxxxxxxxxxxxxxxxx...
KIS_BASE_URL=https://openapi.koreainvestment.com:9443
```

### 3-2. 토큰 발급 테스트

```python
from config.kis_auth import get_access_token

token = get_access_token()
print(f"토큰 발급 성공: {token[:20]}...")
```

---

## 4. 2차시: 데이터 수집 및 검증 실행

### 4-1. 기본 사용 (CSV만 로드)

```python
from src.data.collector import load_and_update

# CSV 파일 로드 (당일 API 수집 없이)
df = load_and_update("005930", collect_today_flag=False)
print(df.shape)
# → (n행, 7열)  [datetime, open, high, low, close, volume, ticker]
```

### 4-2. 당일 데이터 추가 수집 (장 마감 후 실행)

```python
from src.data.collector import load_and_update

# CSV + 당일 API 수집 결합
df = load_and_update("005930", collect_today_flag=True)
```

> ⚠ 당일 수집은 오늘 날짜 데이터만 추가됩니다. 2일 이상 지난 데이터는 추가 수집 불가.

### 4-3. 전체 종목 일괄 로드

```python
from src.data.collector import load_all_tickers

dfs = load_all_tickers(collect_today_flag=False)
# → {"005930": df, "000660": df, ...}
```
python -c "from src.data.collector import load_all_tickers; r=load_all_tickers(collect_today_flag=True); print({k: len(v) for k,v in r.items()})"

### 4-4. 데이터 검증

```python
from src.data.validator import validate, create_sandbox

# 단일 종목 검증 보고서 출력
report = validate(df, ticker="005930")

# 빠른 프로토타이핑용 sandbox 생성 (최근 3개월)
sandbox = create_sandbox("005930", months=3)
```

---

## 5. 3차시: 탐색적 데이터 분석 실행

### 5-1. 전체 EDA (권장)

```python
# Jupyter 셀에서 실행
import sys
sys.path.insert(0, "..")
exec(open("../notebooks/02_eda.py").read())

run_full_eda("005930")
# → reports/figures/ 에 시각화 7종 저장
```

### 5-2. 개별 분석 실행

```python
# 로드
df = load_data("005930")
df = compute_log_returns(df)

# 기술 통계
print_basic_stats(df)

# 시각화 개별 실행
plot_return_histogram(df, ticker="005930")
plot_qq(df, ticker="005930")
plot_rolling_volatility(df, ticker="005930")

# Volatility Clustering 검증
result = test_volatility_clustering(df, ticker="005930")
# → lb_pval_lag20 < 0.05 이면 Clustering 존재 확인

# Leverage Effect
plot_leverage_effect(df, ticker="005930")
```

### 5-3. 결과 파일 확인

```
kium-p/reports/figures/
  005930_hist.png       ← 수익률 히스토그램
  005930_qq.png         ← Q-Q Plot
  005930_boxplot_hour.png  ← 시간대별 박스플롯
  005930_rolling_vol.png   ← Rolling Volatility
  005930_acf_pacf.png      ← ACF/PACF (r_t 및 r_t²)
  005930_vol_clustering.png ← Volatility Clustering 시각화
  005930_leverage.png       ← Leverage Effect
```

---

## 6. 4차시: 특성공학 파이프라인 실행

### 6-1. 전체 파이프라인 (권장)

```python
import sys
sys.path.insert(0, "..")
exec(open("../notebooks/03_feature_engineering.py").read())

# 전체 파이프라인 한 번에 실행
result = run_feature_pipeline("005930")

X_train = result["X_train"]   # shape: (n, 60, 15)
y_train = result["y_train"]   # shape: (n, 1)
X_val   = result["X_val"]
y_val   = result["y_val"]
X_test  = result["X_test"]
y_test  = result["y_test"]
```

### 6-2. 단계별 실행

```python
from notebooks.feature_pipeline import (
    load_clean_data, build_features,
    temporal_split, scale_features,
    create_all_windows, load_windows
)

# 1. 데이터 로드
df = load_clean_data("005930")

# 2. 15개 피처 생성
df_feat = build_features(df)
feat_cols = [c for c in df_feat.columns if c.startswith("F")]

# 3. 분할 (7:2:1)
df_train, df_val, df_test = temporal_split(df_feat)

# 4. 스케일링 (train fit → val/test transform)
df_train, df_val, df_test, scaler = scale_features(
    df_train, df_val, df_test, feat_cols
)

# 5. Window 생성 (기본: window=60봉, horizon=1봉)
splits = create_all_windows(df_train, df_val, df_test, feat_cols)

# 6. 다음 번 로드
splits = load_windows()
```

### 6-3. 생성된 15개 피처 설명

| 번호 | 피처명 | 설명 | 그룹 |
|---|---|---|---|
| F01 | log_ret | 로그수익률 r_t = ln(P_t/P_{t-1}) | 수익률 |
| F02 | log_ret_lag1 | 1봉 전 로그수익률 | 수익률 |
| F03 | log_ret_lag5 | 5봉 전 로그수익률 | 수익률 |
| F04 | log_ret_lag12 | 12봉 전 (~1시간) | 수익률 |
| F05 | log_ret_sq | 수익률 제곱 r_t² | 수익률 |
| F06 | roll_vol_20 | Rolling σ 20봉 | 변동성 |
| F07 | roll_vol_60 | Rolling σ 60봉 | 변동성 |
| F08 | vol_ratio | 단기/장기 변동성 비율 | 변동성 |
| F09 | leverage_proxy | Leverage Effect 프록시 | 변동성 |
| F10 | bar_range | 봉 내 변동폭 | 가격구조 |
| F11 | upper_shadow | 위꼬리 비율 | 가격구조 |
| F12 | lower_shadow | 아래꼬리 비율 | 가격구조 |
| F13 | vol_log | 로그 거래량 | 거래량 |
| F14 | vol_change | 거래량 변화율 | 거래량 |
| F15 | time_sin | 장내 시간 sin 인코딩 | 시간 |

---

## 7. API 제약 안내 및 대응 전략

### 실제 확인된 제약 사항

| API | 제약 | 결과 |
|---|---|---|
| 주식일별분봉조회 (FHKST03010230) | FID_INPUT_DATE_1 날짜 지정 가능하나 시스템상 당일 데이터만 반환 | 날짜 루프 과거 수집 **불가** |
| 주식당일분봉조회 (FHKST03010200) | 당일 데이터만 제공, 1회 30건 | 당일 5분봉 **가능** |

### 데이터 수집 전략 (제약 대응)

```
과거 학습 데이터 (메인)
  └── 기 보유 CSV 파일 → load_from_csv()
  
당일 추가 수집 (검증·점진 업데이트)
  └── KIS 당일분봉 API → collect_today()
  └── 매 거래일 장 마감 후 1회 실행 권장

결합
  └── load_and_update() → 중복 제거 → Parquet 저장
```

### 일별 업데이트 스케줄 (선택)

```bash
# 매 거래일 16:00 이후 실행 (cron 등록 예시)
# 0 16 * * 1-5  python scripts/collect_all.py --today-only
python scripts/collect_all.py --today-only
```

---

## 8. 자주 발생하는 오류와 해결법

### ❌ CSV 파일을 찾을 수 없습니다

```
[005930] CSV 파일을 찾을 수 없습니다.
  다음 위치에 CSV를 배치해 주세요:
  · data/raw/005930.csv
```

**해결:** `data/raw/005930.csv` 경로에 파일을 배치합니다.

---

### ❌ 매핑되지 않은 컬럼

```
'날짜'  → ❌ 미매핑 → '날짜' 추가 필요
```

**해결:** `src/data/collector.py` 상단 `_CSV_COLUMN_MAP`에 추가합니다.

```python
_CSV_COLUMN_MAP = {
    ...
    "날짜": "date",   # ← 추가
    "시간": "time",   # ← 추가
    ...
}
```

---

### ❌ API 토큰 오류

```
RuntimeError: API 오류 [EGW00123]: 접근토큰 발급 실패
```

**해결:** `.env` 파일에 올바른 `KIS_APP_KEY`, `KIS_APP_SECRET`을 입력했는지 확인합니다.

---

### ❌ 당일 수집 시 빈 결과

```
[005930] 당일 데이터 없음 (장 마감 여부 확인)
```

**해결:** 주식 시장 운영시간(09:00~15:30) 이후 또는 공휴일이 아닌지 확인합니다.  
장 마감 후 15:30~16:00 사이에 수집하면 당일 전체 봉이 반환됩니다.

---

### ❌ Scaler 파일 없음

```
FileNotFoundError: Scaler 파일 없음: data/features/minmax_scaler.pkl
```

**해결:** `run_feature_pipeline()`을 먼저 실행하여 Scaler를 생성합니다.

---

## 9. 디렉토리 구조

```
kium-p/
│
├── .env.example              ← API KEY 템플릿 (실제 .env는 .gitignore)
├── .gitignore
├── README.md
├── USAGE_GUIDE.md            ← 이 파일 (사용설명서)
├── requirements.txt
│
├── config/
│   ├── settings.py           ← 전역 상수 (종목 리스트, 경로 등)
│   └── kis_auth.py           ← KIS API 토큰 발급 유틸
│
├── data/
│   ├── raw/                  ← CSV 파일 배치 위치 ← 여기에 CSV 넣기
│   │   ├── 005930.csv
│   │   └── ...
│   ├── cleaned/              ← 전처리 후 Parquet
│   ├── features/             ← 스케일된 데이터, Scaler pkl
│   └── splits/               ← train/val/test.parquet, windows.npz
│
├── notebooks/
│   ├── 01_data_collection.ipynb   ← 2차시
│   ├── 02_eda.py                  ← 3차시 EDA
│   ├── 03_feature_engineering.py  ← 4차시 특성공학
│   └── ...
│
├── src/
│   ├── data/
│   │   ├── collector.py  ← CSV 로드 + 당일 API 수집
│   │   ├── validator.py  ← 데이터 검증 + sandbox 생성
│   │   └── cleaner.py    ← 6단계 클리닝 파이프라인
│   └── ...
│
└── reports/
    └── figures/          ← EDA 시각화 PNG 저장 위치
```

---

*최종 수정: 2026-03-23 | kium-p v0.2*
