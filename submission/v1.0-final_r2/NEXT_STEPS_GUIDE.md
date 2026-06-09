# kium-p 6~15차시 실행 가이드

> 이 문서는 사용자가 정리한 최종 계획을 기준으로, 6차시부터 15차시까지의 작업 흐름과 현재 저장소의 실제 파일 위치를 연결한 후속 로드맵입니다.

---

## 1. 현재 기준 파일 배치

```text
kium-p/
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── features/
│   ├── splits/
│   └── models/
├── notebooks/
├── src/
└── reports/
```

핵심 결과물은 다음 위치에 있습니다.

- 5차시 기준 Baseline: `data/models/lstm_point_best.pt`
- 6~7차시 분위수 결과: `data/models/lstm_q50_best.pt`, `data/models/lstm_multiq_best.pt`, `data/models/week67_results.pkl`
- 8차시 최적화 결과: `data/models/optuna_best.pt`, `data/models/optuna_study.pkl`
- 9차시 평가 결과: `data/models/week9_results.pkl`
- 10차시 TFT 결과: `data/models/tft_best.pt`
- 11차시 SHAP 결과: `data/models/shap_values.npy`, `data/models/shap_feat_importance.npy`
- 12차시 백테스트 결과: `data/models/backtest_results.pkl`
- 13차시 종합 비교: `data/models/final_compare.pkl`
- 시각화 결과: `reports/figures/`

---

## 2. 프로젝트 킥오프

목표:

- KOSPI200 종목 선정, 데이터 소스 확정
- 연구질문과 가설 설정
- 학습용 논문 3편 선별
- 개발환경 구축(PyTorch, Jupyter, Git)

현재 반영 상태:

- KOSPI 상위 10개 종목으로 데이터 소스 확정
- DeepAR, TFT, Regression Quantiles 관련 논문을 참고 문헌으로 정리
- `config/`, `src/`, `notebooks/` 구조로 개발환경 분리

---

## 3. 데이터 수집

목표:

- KRX/Yahoo Finance 5분봉 1년치 다운로드
- VIX, 환율, 거래량 외생변수 수집
- 데이터 검증 및 샘플링

현재 반영 상태:

- 메인 학습 데이터는 `data/raw/kospi_top10_2025.csv`
- KIS 당일 수집은 보조 기능으로 남겨둠
- 샘플링과 검증은 `src/data/collector.py`, `src/data/validator.py`에서 수행

---

## 4. 탐색적 분석

목표:

- 결측치와 이상치 확인
- 분포 시각화
- 상관관계 히트맵과 ACF/PACF 분석
- Volatility Clustering 검증

현재 반영 상태:

- 3차시 EDA는 `notebooks/02_eda.py`
- 결과 그림은 `reports/figures/`
- 변동성, 자기상관, 레버리지 효과 분석을 지원하도록 구성

---

## 5. 데이터 전처리

목표:

- Long-return 계산
- 특성공학 15개 생성
- MinMaxScaler 적용
- 시계열 Window 생성
- train/val/test 분할(7:2:1)

현재 반영 상태:

- `notebooks/03_feature_engineering.py`
- `data/features/minmax_scaler.pkl`
- `data/splits/windows.npz`
- `data/splits/train.parquet`, `val.parquet`, `test.parquet`

---

## 6. Baseline1 구현

목표:

- LSTM Point Prediction(MSE 손실)
- 하이퍼파라미터 탐색(LR, Layer)
- MAE, RMSE 기준 성능 측정

현재 반영 상태:

- `notebooks/04_baseline_lstm.py`
- `src/models/lstm_point.py`
- `src/losses/mse_loss.py`
- `data/models/lstm_point_best.pt`
- `data/models/train_history.npy`

---

## 7. Baseline2 구현

목표:

- Quantile LSTM(τ=0.5 단일)
- Pinball Loss 직접 구현 검증
- Baseline1과 성능 비교

현재 반영 상태:

- `notebooks/05_quantile_lstm.py`
- `src/losses/pinball.py`
- `src/models/lstm_quantile.py`
- `data/models/lstm_q50_best.pt`
- `data/models/week67_results.pkl`

---

## 8. 다분위수 확장

목표:

- Multi-Quantile Head(5개 분위수)
- 다중 타겟 Pinball Loss 구현
- 예측구간 시각화(95% CI)

현재 반영 상태:

- `notebooks/05_quantile_lstm.py` 확장
- `data/models/lstm_multiq_best.pt`
- `reports/figures/7th_001_training_curve.png` 등 7차시 결과 그림

---

## 9. 성능 최적화1

목표:

- Dropout, LayerNorm, Residual 추가
- Early Stopping, Learning Rate Scheduler
- 하이퍼파라미터 튜닝(Optuna)

현재 반영 상태:

- `notebooks/06_optimization.py`
- `data/models/optuna_best.pt`
- `data/models/optuna_study.pkl`
- `reports/figures/8th_001_optuna_history.png` 등

---

## 10. 평가 지표 구현

목표:

- CRPS 함수 직접 구현
- 5개 분위수 Pinball Loss 평균
- 95% VaR Coverage 계산

현재 반영 상태:

- `notebooks/07_evaluation.py`
- `src/losses/crps.py`
- `src/evaluation/quantile_metrics.py`
- `data/models/week9_results.pkl`

---

## 11. 고급모델 도전

목표:

- Temporal Fusion Transformer 시도
- Attention + Gating 메커니즘 학습
- Static/Categorical 변수 통합

현재 반영 상태:

- `notebooks/08_tft.py`
- `src/models/tft.py`
- `data/models/tft_best.pt`
- `reports/figures/10차시 그림은 후속 정리 대상`

---

## 12. 특성 중요도 분석

목표:

- SHAP 값 계산(LSTM vs TFT)
- 외생변수 기여도 순위화
- Volatility Cluster 특성 검증

현재 반영 상태:

- `notebooks/09_shap.py`
- `data/models/shap_values.npy`
- `data/models/shap_feat_importance.npy`
- `reports/figures/11th_001_shap_importance.png`

---

## 13. VaR 백테스트

목표:

- 실제 vs 95% VaR 비교 플롯
- Kupiec Test(p-value > 0.05 목표)
- 포트폴리오 손실 시뮬레이션

현재 반영 상태:

- `notebooks/10_backtest.py`
- `src/evaluation/backtest.py`
- `data/models/backtest_results.pkl`
- `reports/figures/12th_001_var_backtest.png`

---

## 14. 결과 통합

목표:

- 최종 성능표 완성(5개 모델 비교)
- 논문 수준 시각화 10개 제작
- 실패사례와 개선과정 정리

현재 반영 상태:

- `notebooks/11_final_compare.py`
- `data/models/final_compare.pkl`
- `reports/figures/13th_001_final_table.png`
- `reports/figures/13th_002_radar_chart.png`
- `reports/figures/13th_003_metric_bars.png`
- `reports/figures/13th_004_params_vs_crps.png`
- `reports/figures/13th_005_mae_improvement.png`

---

## 15. 보고서 작성과 발표 준비

목표:

- 수학적 이론유도
- 코드/결과 완벽 연동
- PPT 자료와 실시간 데모
- GitHub 완전 정리
- 최종 리허설 및 피드백 반영

현재 반영 상태:

- README와 사용설명서 정리 완료
- 제출용 압축본은 `kium-p_v1.0-final_submission.zip`

---

## 16. Colab 및 백업

권장 백업 파일:

1. `data/splits/windows.npz`
2. `data/models/lstm_point_best.pt`
3. `data/models/week67_results.pkl`
4. `data/models/week9_results.pkl`
5. `data/models/tft_best.pt`
6. `data/models/backtest_results.pkl`
7. `data/models/final_compare.pkl`

Colab 복원 예시:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/kium-p-data/data .
```

---

최종 수정: 2026-06-10 | kium-p v1.0-final