"""
config/settings.py
프로젝트 전역 상수 설정
"""

from pathlib import Path

# ── 경로 ────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT_DIR / "data" / "raw"
DATA_CLEAN = ROOT_DIR / "data" / "cleaned"
DATA_FEAT  = ROOT_DIR / "data" / "features"
DATA_SPLIT = ROOT_DIR / "data" / "splits"
REPORT_DIR = ROOT_DIR / "reports"
FIG_DIR    = REPORT_DIR / "figures"

for _p in [DATA_RAW, DATA_CLEAN, DATA_FEAT, DATA_SPLIT, FIG_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ── 분석 대상 종목 (kospi_top10_2025.csv 기준) ────────────────
# CSV 파일: kospi_top10_2025.csv (일봉, 2025-01-02 ~ 2025-12-30)
# Ticker 형식: {code}.KS → 내부적으로 .KS 제거 후 6자리 코드 사용
TICKERS = {
    "005930": "삼성전자",          # Samsung Electronics
    "000660": "SK하이닉스",        # SK Hynix
    "373220": "LG에너지솔루션",    # LG Energy Solution
    "207940": "삼성바이오로직스",   # Samsung Biologics
    "005380": "현대차",            # Hyundai Motor
    "000270": "기아",              # Kia
    "068270": "셀트리온",          # Celltrion
    "105560": "KB금융",            # KB Financial Group
    "005490": "POSCO홀딩스",       # POSCO Holdings
    "035420": "NAVER",             # NAVER
}

# CSV 파일명 (data/raw/ 아래 배치)
CSV_FILENAME = "kospi_top10_2025.csv"

# ── 수집 기간 ────────────────────────────────────────────────
COLLECT_START = "20250101"   # YYYYMMDD
COLLECT_END   = "20260317"   # YYYYMMDD

# ── 장 운영 시간 (KST) ────────────────────────────────────────
MARKET_OPEN  = "090000"      # HHMMSS
MARKET_CLOSE = "153000"      # HHMMSS

# ── API 호출 제한 ─────────────────────────────────────────────
API_CALL_INTERVAL    = 0.5   # 초 (rate limit 여유 확보)
MAX_RECORDS_PER_CALL = 120   # 일별분봉조회 1회 최대 건수

# ── 데이터 종류 ───────────────────────────────────────────────
# kospi_top10_2025.csv는 일봉(daily) 데이터
DATA_TYPE    = "daily"       # "daily" | "intraday"
BARS_PER_DAY = 1             # 일봉=1, 5분봉=78

# ── 분할 비율 ────────────────────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2
TEST_RATIO  = 0.1

# ── 모델 하이퍼파라미터 기본값 ───────────────────────────────
QUANTILES   = [0.05, 0.25, 0.50, 0.75, 0.95]
# 일봉 기준: Window 20일 = 약 1개월 거래일
# (5분봉 전환 시 → 20 × 78 = 1,560봉으로 변경 예정)
WINDOW_SIZE = 20             # 입력 시퀀스 길이 (일봉 20일)
HORIZON     = 1              # 예측 스텝 (1일 앞)
