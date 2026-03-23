# kium-p

> **금융 자산 수익률의 동적 위험 예측을 위한 다분위수 확률 시계열 예측**  
> 2026학년도 1학기 더 KIUM학기제 | 전공심화형  
> 김도희 · 20201889 · 스마트보안학과 | 지도교수: 김현성

---

## 프로젝트 개요

단일 점 예측(Point Prediction)의 한계를 넘어, KOSPI 상위 10개 종목의 일봉 수익률 데이터에  
**LSTM 기반 Quantile Regression**을 적용하여 다분위수(5% / 25% / 50% / 75% / 95%)를 동시에 예측합니다.  
예측된 분위수를 기반으로 **동적 VaR(95%)** 를 추정하고 포트폴리오 리스크 관리에 응용하는 것이 최종 목표입니다.

### 정량적 목표

| 지표 | 목표값 |
|---|---|
| Pinball Loss | 기존 LSTM Point Prediction 대비 **15% 이하** |
| CRPS | **0.08 이하** |
| 95% VaR Coverage | **92 ~ 98%** |

---

## 데이터

| 항목 | 내용 |
|---|---|
| 파일 | `kospi_top10_2025.csv` |
| 형태 | 일봉(Daily) OHLCV + Adj Close |
| 기간 | 2025-01-02 ~ 2025-12-30 |
| 종목 | 10개 (아래 표 참조) |
| 행 수 | 종목당 241행 (추석 공휴일 2025-09-19 자동 제거) |

### 분석 대상 종목

| 종목코드 | 종목명 |
|---|---|
| 005930 | 삼성전자 |
| 000660 | SK하이닉스 |
| 373220 | LG에너지솔루션 |
| 207940 | 삼성바이오로직스 |
| 005380 | 현대차 |
| 000270 | 기아 |
| 068270 | 셀트리온 |
| 105560 | KB금융 |
| 005490 | POSCO홀딩스 |
| 035420 | NAVER |

> **KIS API 제약**: 날짜 기반 과거 분봉 수집 불가 (당일 데이터만 제공).  
> 현재는 CSV 일봉 데이터로 파이프라인을 구축하며, 5분봉 데이터 확보 시 `WINDOW_SIZE` 변경만으로 전환 가능.

---

## 디렉토리 구조

```
kium-p/
│
├── .env.example                   # API KEY 템플릿 (.env는 .gitignore)
├── .gitignore
├── README.md
├── USAGE_GUIDE.md                 # 단계별 실행 사용설명서
├── requirements.txt               # 패키지 버전 고정
│
├── config/
│   ├── settings.py                # 전역 상수 (종목·경로·Window·분할 비율)
│   └── kis_auth.py                # KIS Open API 토큰 발급·캐싱
│
├── data/
│   ├── raw/
│   │   └── kospi_top10_2025.csv   # ← 원본 CSV 파일 배치 위치
│   ├── cleaned/
│   │   └── {종목코드}_merged.parquet   # 클리닝 완료 (10개 종목)
│   ├── features/
│   │   └── minmax_scaler.pkl      # 학습된 MinMaxScaler
│   └── splits/
│       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       └── windows.npz            # 슬라이딩 Window X·y 배열
│
├── notebooks/
│   ├── 02_eda.py                  # 3차시: EDA·분포 시각화·Clustering 검증
│   ├── 03_feature_engineering.py  # 4차시: 15개 피처·스케일·Window 생성
│   ├── 04_baseline_lstm.py        # 5차시: LSTM Point Prediction (예정)
│   ├── 05_quantile_lstm.py        # 6~7차시: Multi-Quantile LSTM (예정)
│   ├── 06_optimization.py         # 8차시: Optuna 튜닝 (예정)
│   ├── 07_evaluation.py           # 9차시: CRPS·VaR Coverage (예정)
│   └── 08_backtest.py             # 12차시: VaR 백테스트 (예정)
│
├── src/
│   ├── data/
│   │   ├── collector.py           # CSV 로드 + KIS 당일 분봉 수집 + 결합
│   │   ├── cleaner.py             # 6단계 클리닝 (일봉/분봉 자동 감지)
│   │   └── validator.py           # 5항목 검증 보고서 + sandbox 생성
│   ├── models/
│   │   ├── lstm_point.py          # Baseline LSTM MSE (예정)
│   │   ├── lstm_quantile.py       # Multi-Quantile LSTM (예정)
│   │   └── tft.py                 # Temporal Fusion Transformer (예정)
│   ├── losses/
│   │   ├── pinball.py             # Quantile(Pinball) Loss PyTorch 구현 (예정)
│   │   └── crps.py                # CRPS 함수 (예정)
│   └── evaluation/
│       ├── metrics.py             # VaR Coverage·Kupiec Test (예정)
│       └── backtest.py            # 포트폴리오 백테스트 (예정)
│
└── reports/
    ├── eda_summary.md
    └── figures/                   # EDA 시각화 PNG 저장 위치
```

---

## 환경 구축

```bash
# 1. 가상환경
conda create -n kium_quant python=3.10
conda activate kium_quant

# 2. 패키지 설치
pip install -r requirements.txt

# 3. PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. GPU 확인
python -c "import torch; print(torch.cuda.is_available())"   # → True

# 5. API KEY 설정
cp .env.example .env
# .env 파일에 KIS_APP_KEY, KIS_APP_SECRET 입력
```

---

## 빠른 시작 (Quick Start)

### Step 1 — 데이터 로드

```python
from src.data.collector import load_all_tickers

# kospi_top10_2025.csv → 10종목 Parquet 저장
dfs = load_all_tickers(collect_today_flag=False)
```

### Step 2 — 클리닝

```python
from src.data.collector import load_from_csv
from src.data.cleaner import clean_and_save

df = load_from_csv("005930")
df_clean = clean_and_save(df, "005930")
```

### Step 3 — EDA

```python
# notebooks/02_eda.py 실행
from notebooks.eda import run_full_eda
run_full_eda("005930")
# → reports/figures/ 에 시각화 저장
```

### Step 4 — 특성공학 + Window 생성

```python
# notebooks/03_feature_engineering.py 실행
from notebooks.feature_pipeline import run_feature_pipeline
result = run_feature_pipeline("005930")

X_train = result["X_train"]  # shape: (n, 20, 15)
y_train = result["y_train"]  # shape: (n, 1)
```

---

## 생성 피처 목록 (15개)

| # | 피처명 | 설명 | 그룹 |
|---|---|---|---|
| F01 | log_ret | 로그수익률 r_t | 수익률 |
| F02 | log_ret_lag1 | 1일 전 수익률 | 수익률 |
| F03 | log_ret_lag5 | 5일 전 수익률 | 수익률 |
| F04 | log_ret_lag12 | 12일 전 수익률 | 수익률 |
| F05 | log_ret_sq | 수익률 제곱 r_t² | 수익률 |
| F06 | roll_vol_20 | Rolling σ 20일 | 변동성 |
| F07 | roll_vol_60 | Rolling σ 60일 | 변동성 |
| F08 | vol_ratio | 단기/장기 변동성 비율 | 변동성 |
| F09 | leverage_proxy | Leverage Effect 프록시 | 변동성 |
| F10 | bar_range | (high-low)/close | 가격구조 |
| F11 | upper_shadow | 위꼬리 비율 | 가격구조 |
| F12 | lower_shadow | 아래꼬리 비율 | 가격구조 |
| F13 | vol_log | 로그 거래량 | 거래량 |
| F14 | vol_change | 거래량 변화율 | 거래량 |
| F15 | time_sin | 장내 시간 sin 인코딩 | 시간 |

---

## 차시별 진행 현황

| 차시 | 내용 | 상태 |
|---|---|---|
| 1차시 | 프로젝트 킥오프, GitHub 저장소, 환경구축 | ✅ 완료 |
| 2차시 | CSV 데이터 수집·통합·검증 | ✅ 완료 |
| 3차시 | EDA, 결측치 처리, Volatility Clustering 검증 | ✅ 완료 |
| 4차시 | 특성공학 15개, MinMaxScaler, Window Dataset | ✅ 완료 |
| 5차시 | LSTM Point Prediction (MSE 손실) | 🔜 예정 |
| 6차시 | Quantile LSTM (τ=0.5 단일), Pinball Loss 구현 | 🔜 예정 |
| 7차시 | Multi-Quantile Head (5개 분위수), 예측구간 시각화 | 🔜 예정 |
| 8차시 | Dropout·LayerNorm·Residual, Optuna 튜닝 | 🔜 예정 |
| 9차시 | CRPS 함수 구현, 95% VaR Coverage 계산 | 🔜 예정 |
| 10차시 | Temporal Fusion Transformer 시도 | 🔜 예정 |
| 11차시 | SHAP 특성 중요도 분석 | 🔜 예정 |
| 12차시 | VaR 백테스트, Kupiec Test | 🔜 예정 |
| 13차시 | 결과 통합, 5개 모델 비교 | 🔜 예정 |
| 14차시 | 보고서 작성 (수학적 이론유도 + 코드 연동) | 🔜 예정 |
| 15차시 | PPT + 실시간 데모, GitHub 완전 정리 | 🔜 예정 |

---

## 참고 논문

1. Salinas et al. (2020) — *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks*, IJMLR
2. Lim et al. (2021) — *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*, IJF
3. Koenker & Bassett (1978) — *Regression Quantiles*, Econometrica

---

*최종 수정: 2026-03-23 | kium-p v0.3*
