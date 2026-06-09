# kium-p

> 금융 자산 수익률의 동적 위험 예측을 위한 다분위수 확률 시계열 예측
> 2026학년도 1학기 더 KIUM학기제 | 김도희 · 20201889 · 스마트보안학과

---

## 프로젝트 개요

kium-p는 KOSPI 상위 10개 종목의 일봉 OHLCV 데이터를 바탕으로 로그수익률, 변동성, 가격구조, 거래량, 시간 피처를 생성하고, LSTM 기반 예측 모델로 수익률과 위험지표를 추정하는 프로젝트입니다.

이 문서는 1주차부터 15주차까지의 진행 흐름을 하나의 README 안에 정리한 최종 요약본입니다.

---

## 1~15주차 로드맵

| 주차 | 핵심 내용 | 주요 파일 | 결과물 |
|---|---|---|---|
| 1주차 | 프로젝트 킥오프, KOSPI200 종목 선정, 데이터 소스 확정, 연구질문/가설 설정, 논문 3편 선별, 개발환경 구축 | `config/settings.py`, `config/kis_auth.py`, `requirements.txt` | 개발환경, 종목 리스트, 연구 방향 정리 |
| 2주차 | 데이터 수집, CSV 배치, 데이터 검증, 샘플링 | `src/data/collector.py`, `src/data/validator.py`, `download_data_kis.py` | `data/raw/kospi_top10_2025.csv`, 검증 로그 |
| 3주차 | 탐색적 분석, 결측치·이상치 확인, 분포 시각화, 상관관계 히트맵, ACF/PACF, Volatility Clustering 검증 | `src/data/cleaner.py`, `notebooks/02_eda.py` | `reports/figures/5th_*.png`, `6th_*.png` |
| 4주차 | Long-return 계산, 특성공학 15개 생성, MinMaxScaler, 시계열 Window 생성, train/val/test 분할 | `notebooks/03_feature_engineering.py` | `data/cleaned/*_clean.parquet`, `data/features/minmax_scaler.pkl`, `data/splits/windows.npz` |
| 5주차 | Baseline1 구현, LSTM Point Prediction, 하이퍼파라미터 탐색, MAE/RMSE 평가 | `src/models/lstm_point.py`, `src/losses/mse_loss.py`, `notebooks/04_baseline_lstm.py` | `data/models/lstm_point_best.pt`, `data/models/train_history.npy`, `data/models/week67_results.pkl` |
| 6주차 | Baseline2 구현, Quantile LSTM(τ=0.5), Pinball Loss 직접 구현 검증 | `src/models/lstm_quantile.py`, `src/losses/pinball.py`, `notebooks/05_quantile_lstm.py` | `data/models/lstm_q50_best.pt`, `data/models/week67_results.pkl` |
| 7주차 | Multi-Quantile Head, 5개 분위수 동시 예측, 예측구간 시각화(95% CI) | `notebooks/05_quantile_lstm.py` | `data/models/lstm_multiq_best.pt`, `reports/figures/7th_*.png` |
| 8주차 | 성능 최적화, Dropout/LayerNorm/Residual, Early Stopping, Scheduler, Optuna 튜닝 | `notebooks/06_optimization.py` | `data/models/optuna_best.pt`, `data/models/optuna_study.pkl`, `reports/figures/8th_*.png` |
| 9주차 | 평가 지표 구현, CRPS 함수, Pinball 평균, 95% VaR Coverage 계산 | `src/losses/crps.py`, `src/evaluation/quantile_metrics.py`, `notebooks/07_evaluation.py` | `data/models/week9_results.pkl`, `reports/figures/9th_*.png` |
| 10주차 | Temporal Fusion Transformer 시도, Attention + Gating, Static/Categorical 변수 통합 | `src/models/tft.py`, `notebooks/08_tft.py` | `data/models/tft_best.pt` |
| 11주차 | SHAP 값 계산, LSTM vs TFT 특성 중요도 비교, Volatility Cluster 특성 검증 | `notebooks/09_shap.py` | `data/models/shap_values.npy`, `data/models/shap_feat_importance.npy`, `reports/figures/11th_*.png` |
| 12주차 | VaR 백테스트, 실제 vs 95% VaR 비교, Kupiec Test, 손실 시뮬레이션 | `src/evaluation/backtest.py`, `notebooks/10_backtest.py` | `data/models/backtest_results.pkl`, `reports/figures/12th_*.png` |
| 13주차 | 결과 통합, 5개 모델 비교, 논문 수준 시각화 제작 | `notebooks/11_final_compare.py` | `data/models/final_compare.pkl`, `reports/figures/13th_*.png` |
| 14주차 | 보고서 작성, 수학적 이론 유도, 코드/결과 연동 | `2026학년도 1학기 더KIUM학기제 결과보고서.*`, `2026학년도 1학기 더KIUM학기제 활동보고서.hwp` | 최종 보고서/활동보고서 정리 |
| 15주차 | PPT 자료, 실시간 데모, GitHub 최종 정리, 리허설 및 피드백 반영 | `README.md`, 제출 zip | `kium-p_v1.0-final_submission_20260610.zip` |

---

## 핵심 결과 요약

| 항목 | 내용 |
|---|---|
| 원천 데이터 | `data/raw/kospi_top10_2025.csv` |
| 데이터 형태 | KOSPI 10종목 일봉 OHLCV + Adj Close |
| 기간 | 2025-01-02 ~ 2025-12-30 |
| 대상 종목 수 | 10개 |
| 전처리 결과 | 종목별 `data/cleaned/*_clean.parquet` |
| 피처 수 | 15개 |
| 입력 Window | `WINDOW_SIZE=20`, `HORIZON=1` |
| 분할 데이터 | `data/splits/windows.npz` |
| Baseline 모델 | `data/models/lstm_point_best.pt` |

### 대표 수치

| 모델 | 지표 | 결과 |
|---|---|---|
| Baseline LSTM | Val MAE | 0.2248 |
| Baseline LSTM | Val RMSE | 0.6177 |
| Multi-Q LSTM | CRPS | 0.1742 |
| Multi-Q LSTM | PICP | 89.11% |
| Optuna LSTM | CRPS | 0.1539 |
| Optuna LSTM | PICP | 93.25% |
| TFT | CRPS | 0.1827 |
| TFT | PICP | 94.34% |

---

## Quick Start

### 1. 환경 준비

```bash
conda create -n kium_quant python=3.10
conda activate kium_quant
pip install -r requirements.txt
```

### 2. 데이터 배치

`data/raw/` 아래에 다음 파일을 둡니다.

```text
data/raw/kospi_top10_2025.csv
```

### 3. 실행 순서

```bash
python notebooks/02_eda.py
python notebooks/03_feature_engineering.py
python notebooks/04_baseline_lstm.py
python notebooks/05_quantile_lstm.py
python notebooks/06_optimization.py
python notebooks/07_evaluation.py
python notebooks/08_tft.py
python notebooks/09_shap.py
python notebooks/10_backtest.py
python notebooks/11_final_compare.py
```

### 4. 결과 확인

```bash
python -c "import numpy as np; d=np.load('data/splits/windows.npz'); print(d['X_train'].shape, d['X_val'].shape, d['X_test'].shape)"
```

정상적으로 생성되면 `windows.npz`, `lstm_point_best.pt`, `train_history.npy`가 `data/` 아래에 저장됩니다.

---

## 코드 구조

```text
kium-p/
├── config/
│   ├── settings.py
│   └── kis_auth.py
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── features/
│   └── splits/
├── notebooks/
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_baseline_lstm.py
│   ├── 05_quantile_lstm.py
│   ├── 06_optimization.py
│   ├── 07_evaluation.py
│   ├── 08_tft.py
│   ├── 09_shap.py
│   ├── 10_backtest.py
│   └── 11_final_compare.py
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── losses/
│   └── models/
└── reports/
    └── figures/
```

---

## 참고 산출물

- `data/models/lstm_point_best.pt`
- `data/models/lstm_q50_best.pt`
- `data/models/lstm_multiq_best.pt`
- `data/models/optuna_best.pt`
- `data/models/tft_best.pt`
- `data/models/shap_values.npy`
- `data/models/backtest_results.pkl`
- `data/models/final_compare.pkl`

---

## 참고 문서

- `USAGE_GUIDE.md` — CSV 배치, 단계별 실행, 오류 해결법
- 제출용 압축본: `kium-p_v1.0-final_submission_20260610.zip`

---

*최종 수정: 2026-06-10 | kium-p v1.0-final*# kium-p v1.0 Final

> 금융 자산 수익률의 동적 위험 예측을 위한 다분위수 확률 시계열 예측
> 2026학년도 1학기 더 KIUM학기제 | 김도희 · 20201889 · 스마트보안학과

---

## 프로젝트 요약

kium-p는 KOSPI 상위 10개 종목의 일봉 OHLCV 데이터를 이용해 로그수익률, 변동성, 가격구조, 거래량, 시간 피처를 생성하고, LSTM 기반 예측 모델로 수익률과 위험지표를 추정하는 프로젝트입니다.


---

## 최종 결과 요약

| 항목 | 내용 |
|---|---|
| 원천 데이터 | `data/raw/kospi_top10_2025.csv` |
| 데이터 형태 | KOSPI 10종목 일봉 OHLCV + Adj Close |
| 기간 | 2025-01-02 ~ 2025-12-30 |
| 대상 종목 수 | 10개 |
| 전처리 결과 | 종목별 `data/cleaned/*_clean.parquet` |
| 특성공학 | 15개 피처, `WINDOW_SIZE=20`, `HORIZON=1` |
| Window 데이터 | `data/splits/windows.npz` |
| Baseline 모델 | `data/models/lstm_point_best.pt` |
| 학습 히스토리 | `data/models/train_history.npy` |

### 핵심 지표

| 지표 | 결과 | 해석 |
|---|---|---|
| Baseline Val MAE | 0.2248 | 점예측 기준선 |
| Baseline Val RMSE | 0.6177 | 점예측 오차 |
| Multi-Q CRPS | 0.1742 | 확률 예측 결과 |
| Multi-Q PICP | 89.11% | 95% 예측구간 커버리지 |
| Optuna CRPS | 0.1539 | 튜닝 후 개선 결과 |
| Optuna PICP | 93.25% | 커버리지 개선 결과 |
| TFT CRPS | 0.1827 | 최종 비교 대상 |
| TFT PICP | 94.34% | 최종 비교 대상 |

> 현재 기준으로는 CRPS 목표치와 일부 VaR 위험지표가 아직 더 개선이 필요합니다.

---

---

## Quick Start

### 1) 환경 준비

```bash
conda create -n kium_quant python=3.10
conda activate kium_quant
pip install -r requirements.txt
```

### 2) 데이터 배치

`data/raw/` 아래에 다음 파일을 둡니다.

```text
data/raw/kospi_top10_2025.csv
```

KIS API를 함께 쓸 경우 `.env` 파일에 `KIS_APP_KEY`, `KIS_APP_SECRET`을 넣습니다.

### 3) 파이프라인 실행

```bash
python notebooks/02_eda.py
python notebooks/03_feature_engineering.py
python notebooks/04_baseline_lstm.py
```

### 4) 결과 확인

```bash
python -c "import numpy as np; d=np.load('data/splits/windows.npz'); print(d['X_train'].shape, d['X_val'].shape, d['X_test'].shape)"
```

정상적으로 생성되면 `windows.npz`, `lstm_point_best.pt`, `train_history.npy`가 `data/` 아래에 저장됩니다.

---

## 프로젝트 구조

```text
kium-p/
├── config/
│   ├── settings.py
│   └── kis_auth.py
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── features/
│   └── splits/
├── notebooks/
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_baseline_lstm.py
│   ├── 05_quantile_lstm.py
│   ├── 06_optimization.py
│   ├── 07_evaluation.py
│   ├── 08_tft.py
│   ├── 09_shap.py
│   ├── 10_backtest.py
│   └── 11_final_compare.py
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── losses/
│   └── models/
├── reports/
│   └── figures/
├── README.md
├── USAGE_GUIDE.md
└── NEXT_STEPS_GUIDE.md
```

---

## 참고 문서

- `USAGE_GUIDE.md` — CSV 배치, 단계별 실행, 오류 해결법
- `NEXT_STEPS_GUIDE.md` — 6~15차시 실행, Colab 복원, 데이터 백업


---

*최종 수정: 2026-06-10 | kium-p v1.0-final*# kium-p v1.0 Final

> 금융 자산 수익률의 동적 위험 예측을 위한 다분위수 확률 시계열 예측
> 2026학년도 1학기 더 KIUM학기제 | 김도희 · 20201889 · 스마트보안학과

---

## 프로젝트 요약

kium-p는 KOSPI 상위 10개 종목의 일봉 OHLCV 데이터를 이용해 로그수익률, 변동성, 가격구조, 거래량, 시간 피처를 생성하고, LSTM 기반 예측 모델로 수익률과 위험지표를 추정하는 프로젝트입니다.

현재 저장소의 핵심 파이프라인은 1~5차시까지 정리되어 있으며, 데이터 수집·클리닝·특성공학·LSTM Baseline 학습까지 재현할 수 있습니다. 6~15차시는 `NEXT_STEPS_GUIDE.md`에 후속 실행 가이드로 분리했습니다.

---

## 최종 결과 요약

| 항목 | 내용 |
|---|---|
| 원천 데이터 | `data/raw/kospi_top10_2025.csv` |
| 데이터 형태 | KOSPI 10종목 일봉 OHLCV + Adj Close |
| 기간 | 2025-01-02 ~ 2025-12-30 |
| 대상 종목 수 | 10개 |
| 전처리 결과 | 종목별 `data/cleaned/*_clean.parquet` |
| 특성공학 | 15개 피처, `WINDOW_SIZE=20`, `HORIZON=1` |
| Window 데이터 | `data/splits/windows.npz` |
| Baseline 모델 | `data/models/lstm_point_best.pt` |
| 학습 히스토리 | `data/models/train_history.npy` |

### 핵심 지표

| 지표 | 결과 | 해석 |
|---|---|---|
| Baseline Val MAE | 0.2248 | 점예측 기준선 |
| Baseline Val RMSE | 0.6177 | 점예측 오차 |
| Multi-Q CRPS | 0.1742 | 확률 예측 결과 |
| Multi-Q PICP | 89.11% | 95% 예측구간 커버리지 |
| Optuna CRPS | 0.1539 | 튜닝 후 개선 결과 |
| Optuna PICP | 93.25% | 커버리지 개선 결과 |
| TFT CRPS | 0.1827 | 최종 비교 대상 |
| TFT PICP | 94.34% | 최종 비교 대상 |

> 현재 기준으로는 CRPS 목표치와 일부 VaR 위험지표가 아직 더 개선이 필요합니다.

---

---

## Quick Start

### 1) 환경 준비

```bash
conda create -n kium_quant python=3.10
conda activate kium_quant
pip install -r requirements.txt
```

### 2) 데이터 배치

`data/raw/` 아래에 다음 파일을 둡니다.

```text
data/raw/kospi_top10_2025.csv
```

KIS API를 함께 쓸 경우 `.env` 파일에 `KIS_APP_KEY`, `KIS_APP_SECRET`을 넣습니다.

### 3) 파이프라인 실행

```bash
python notebooks/02_eda.py
python notebooks/03_feature_engineering.py
python notebooks/04_baseline_lstm.py
```

### 4) 결과 확인

```bash
python -c "import numpy as np; d=np.load('data/splits/windows.npz'); print(d['X_train'].shape, d['X_val'].shape, d['X_test'].shape)"
```

정상적으로 생성되면 `windows.npz`, `lstm_point_best.pt`, `train_history.npy`가 `data/` 아래에 저장됩니다.

---

## 프로젝트 구조

```text
kium-p/
├── config/
│   ├── settings.py
│   └── kis_auth.py
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── features/
│   └── splits/
├── notebooks/
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_baseline_lstm.py
│   ├── 05_quantile_lstm.py
│   ├── 06_optimization.py
│   ├── 07_evaluation.py
│   ├── 08_tft.py
│   ├── 09_shap.py
│   ├── 10_backtest.py
│   └── 11_final_compare.py
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── losses/
│   └── models/
├── reports/
│   └── figures/
├── README.md
├── USAGE_GUIDE.md
└── NEXT_STEPS_GUIDE.md
```

---

## 참고 문서

- `USAGE_GUIDE.md` — CSV 배치, 단계별 실행, 오류 해결법
- `NEXT_STEPS_GUIDE.md` — 6~15차시 실행, Colab 복원, 데이터 백업

---

*최종 수정: 2026-06-05 | kium-p v1.0-final*# kium-p

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


## 참고 논문

1. Salinas et al. (2020) — *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks*, IJMLR
2. Lim et al. (2021) — *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*, IJF
- 제출용 압축본: `kium-p_v1.0-final_submission.zip`
3. Koenker & Bassett (1978) — *Regression Quantiles*, Econometrica

---
*최종 수정: 2026-06-10 | kium-p v1.0-final*
*최종 수정: 2026-03-23 | kium-p v0.3*
