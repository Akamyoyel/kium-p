# KIUM 금융시계열 프로젝트

KOSPI200 기반 고빈도(5분봉) 데이터를 활용해 변동성 및 리스크(VaR/ES)를 예측하는 프로젝트입니다.

## 1차시 목표

### 1) KOSPI200 종목 선정 및 데이터 소스 확정
- **종목 유니버스**: KOSPI200 구성종목
- **실험 대상 선정 규칙**:
	- 전체 KOSPI200 중 유동성이 높은 종목 우선
	- `download_data.py`에서 기본 `TOP_N_KOSPI200=20`으로 설정
- **데이터 소스**:
	- 가격/거래량(5분봉): Yahoo Finance (`yfinance`)
	- KOSPI200 구성종목: KRX (`pykrx`)
	- 외생변수: VIX(`^VIX`), 원/달러(`USDKRW=X`), KOSPI200 지수(`^KS200`)

### 2) 연구질문/가설 설정
- **RQ1**: 외생변수(VIX, 환율, 지수)가 KOSPI200 개별 종목 변동성 예측 성능을 개선하는가?
- **RQ2**: 확률 예측 모델(DeepAR, TFT, Deep Quantile 기반)이 점예측 모델 대비 VaR/ES 추정에서 우수한가?
- **RQ3**: 시장 레짐(고변동/저변동)에 따라 모델 성능 차이가 발생하는가?

가설:
- **H1**: 외생변수를 포함한 모델이 미포함 모델보다 VaR/ES 백테스트 성능이 높다.
- **H2**: 분포를 직접 학습하는 모델이 꼬리위험(좌측 tail) 예측에서 우수하다.
- **H3**: 변동성 군집이 강한 구간에서 시계열 딥러닝 모델의 상대적 이점이 커진다.

### 3) 학습 논문 3편
1. **DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks**
2. **Temporal Fusion Transformers for Interpretable Multi-horizon Forecasting**
3. **Forecasting VaR and ES by using deep quantile regression, GANs-based scenario generation, and heterogeneous market hypothesis (Wang et al., 2024)**

### 4) 개발환경 구축 (PyTorch, Jupyter/Colab, Git)

로컬:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Jupyter:
```bash
jupyter lab
```

Google Colab:
- 저장소를 GitHub에 push 후 Colab에서 notebook 열기
- 또는 Colab에서 `!pip install -r requirements.txt` 실행 후 스크립트/노트북 사용

Git 초기 작업:
```bash
git add .
git commit -m "init: phase 1-3 data/eda pipeline"
```

## 2차시 목표: 데이터 수집

- KRX/Yahoo 기반 5분봉 1년치 수집
	- Yahoo 5분봉 제한(약 60일)을 고려해 구간 분할 다운로드 적용
- 외생변수 수집
	- VIX, USDKRW, KOSPI200 지수, 거래량 포함
- 데이터 검증/샘플링
	- 결측치 비율, 중복 인덱스, 관측치 수 요약
	- 결과: `data/validation_summary.csv`

실행:
```bash
python download_data.py
```

KIS Developers API 반복요청 루프 실행(국내주식 5분봉):
```bash
# .env 템플릿 복사 후 값 입력
copy .env.example .env

# .env 편집
# KIS_APP_KEY=발급받은_APP_KEY
# KIS_APP_SECRET=발급받은_APP_SECRET
# KIS_BASE_URL=https://openapi.koreainvestment.com:9443
# KIS_TR_ID=FHKST03010200

python download_data_kis.py --dotenv .env --tickers 005930 000660 035420 --days 365 --max-calls 300
```

매일 누적 수집(권장):
```bash
python download_data_kis.py --dotenv .env --tickers 005930 000660 035420 --days 2 --max-calls 80 --sleep-sec 0.2 --append-existing
```

설명:
- `--append-existing`은 기존 `data/raw/{ticker}_KIS_5m.csv`와 신규 수집분을 병합하고 datetime 중복을 제거합니다.
- 최신 구간만 반복적으로 받아도 파일이 누적되어 자체 히스토리 저장소를 운영할 수 있습니다.
- 누적 없이 이번 실행 결과만 저장하려면 `--no-append-existing`을 사용합니다.

설명:
- `download_data_kis.py`는 KIS `주식당일분봉조회` API를 반복 호출하여 가능한 과거 구간까지 누적합니다.
- API 제약으로 종목/계좌 환경에 따라 1년 전체 커버리지가 제한될 수 있으며, 이 경우 스크립트가 경고를 출력합니다.
- 검증 파일은 `data/validation_summary_kis.csv`로 저장됩니다.
- `.env` 파일은 `.gitignore`로 제외되어 키가 저장소에 커밋되지 않습니다.

과거 5분봉 확보 전략(권장):
- KIS는 최근/당일 구간 수집에 집중하고, 스크립트를 매일 실행하여 내부 CSV를 장기 누적합니다.
- 이미 지난 장기 과거 구간(예: 수개월~수년)은 별도 히스토리 소스(유료 벤더, HTS 내보내기 등)로 백필합니다.
- 최종 학습 데이터는 "백필 데이터 + 매일 KIS 누적 데이터"를 병합해 사용합니다.

생성 산출물:
- 원천 데이터: `data/raw/*.csv`
- 검증 요약: `data/validation_summary.csv`

## 3차시 목표: 탐색적 분석

- 결측치/이상치 확인
- 수익률 분포 시각화
- 상관관계 히트맵
- ACF/PACF 분석(제곱수익률)
- 변동성 군집(Volatility Clustering) 확인

실행:
```bash
python exploratory_analysis.py
```

생성 산출물:
- `data/eda/*_returns_distribution.png`
- `data/eda/*_volatility_clustering.png`
- `data/eda/*_acf_pacf_squared_returns.png`
- `data/eda/correlation_heatmap.png`

## 파일 설명

- `download_data.py`: KOSPI200/외생변수 다운로드 + 검증 요약 저장
- `download_data_kis.py`: KIS API 반복요청 루프 기반 5분봉 수집 + 검증 요약 저장
- `exploratory_analysis.py`: EDA 및 시각화 결과 저장
- `extract_pdf_text.py`: 문서 텍스트 추출용(필요 시 사용)

