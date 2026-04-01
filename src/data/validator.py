"""
src/data/validator.py
────────────────────────────────────────────────────────────────────────────
2차시: 데이터 검증 및 샘플링

■ 검증 항목
  1. 시간축 정합성  : 모든 datetime이 장 운영 시간(09:00~15:30) 내에 있는지
  2. 결측 구간       : 예상 봉 수 대비 실제 봉 수 비율 (결측률)
  3. 가격 이상치     : 1봉 내 ±10% 초과 변동 또는 가격 = 0
  4. OHLC 논리 검증 : low ≤ open, close ≤ high 여부
  5. 거래량 이상     : volume = 0 봉 비율

■ 샘플링 전략
  - 빠른 프로토타이핑용: 삼성전자(005930) 최근 3개월치 → sandbox 저장
  - 모델 개발 완료 후 전체 종목·전체 기간으로 확장
────────────────────────────────────────────────────────────────────────────
"""

import logging
from pathlib import Path
from datetime import time as dtime

import numpy as np
import pandas as pd

from config.settings import DATA_RAW, DATA_CLEAN, TICKERS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── 장 운영 시간 경계 ────────────────────────────────────────
MARKET_START = dtime(9, 0, 0)
MARKET_END   = dtime(15, 30, 0)

# ── 이상치 임계값 ────────────────────────────────────────────
PRICE_SPIKE_THRESHOLD = 0.10   # 1봉 내 고저 변동 10% 초과 = 이상치 후보


def _is_daily_frame(df: pd.DataFrame) -> bool:
    """일봉 여부 자동 판별 (_is_daily 플래그 또는 00:00:00 시각 패턴)."""
    if "_is_daily" in df.columns and df["_is_daily"].astype(bool).any():
        return True

    dt = pd.to_datetime(df["datetime"], errors="coerce")
    dt = dt.dropna()
    if dt.empty:
        return False

    has_intraday_time = ((dt.dt.hour != 0) | (dt.dt.minute != 0) | (dt.dt.second != 0)).any()
    return not has_intraday_time


# ═══════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════════

def load_ticker_data(ticker: str, data_dir: Path = DATA_RAW) -> pd.DataFrame:
    """
    특정 종목 Parquet을 자동 탐색하여 로드.

    탐색 우선순위
    1) data_dir/merged_{ticker}.parquet
    2) data/cleaned/{ticker}_merged.parquet  (collector 기본 저장 형식)
    3) data/cleaned/merged_{ticker}.parquet
    4) data_dir/{ticker}/*.parquet
    """
    merged_candidates = [
        data_dir / f"merged_{ticker}.parquet",
        DATA_CLEAN / f"{ticker}_merged.parquet",
        DATA_CLEAN / f"merged_{ticker}.parquet",
    ]

    for merged in merged_candidates:
        if merged.exists():
            df = pd.read_parquet(merged)
            logger.info(f"[{ticker}] merged 파일 로드: {len(df):,}행 ({merged})")
            return df

    # 일자별 파일 수집
    ticker_dir = data_dir / ticker
    files = sorted(ticker_dir.glob("*.parquet")) if ticker_dir.exists() else []
    if not files:
        logger.warning(
            f"[{ticker}] 수집된 파일 없음\n"
            f"  확인 경로:\n"
            f"  - {data_dir / f'merged_{ticker}.parquet'}\n"
            f"  - {DATA_CLEAN / f'{ticker}_merged.parquet'}\n"
            f"  - {DATA_CLEAN / f'merged_{ticker}.parquet'}\n"
            f"  - {ticker_dir / '*.parquet'}"
        )
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"[{ticker}] {len(files)}개 파일 → 총 {len(df):,}행")
    return df


# ═══════════════════════════════════════════════════════════════
# 2. 검증 보고서 생성
# ═══════════════════════════════════════════════════════════════

def validate(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    DataFrame에 대한 전체 검증 수행.
    결과를 딕셔너리로 반환하고 콘솔에 요약 출력.

    Returns
    -------
    dict  {
        "total_rows"         : int,
        "date_range"         : (start, end),
        "missing_rate"       : float,   # 예상 봉 대비 결측 비율
        "out_of_hours"       : int,     # 장 시간 외 봉 수
        "zero_price"         : int,     # 가격 = 0 봉 수
        "ohlc_violations"    : int,     # low > open or close > high 봉 수
        "price_spikes"       : list,    # 이상치 후보 인덱스
        "zero_volume"        : int,     # 거래량 = 0 봉 수
        "zero_volume_rate"   : float,
    }
    """
    if df.empty:
        return {}

    # datetime 타입 보정
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    total = len(df)
    label = f"[{ticker}]" if ticker else ""

    # ── 0. 데이터 유형 감지 ─────────────────────────────────────
    is_daily = _is_daily_frame(df)
    bars_per_day = 1 if is_daily else 78
    data_type = "daily" if is_daily else "intraday_5min"

    # ── 1. 시간축 검증 ─────────────────────────────────────────
    if is_daily:
        out_hours = 0
    else:
        times = df["datetime"].dt.time
        out_hours = ((times < MARKET_START) | (times > MARKET_END)).sum()

    # ── 2. 결측률 추정 ─────────────────────────────────────────
    # 수집 기간의 영업일 수 × 봉 수/일 = 예상 봉 수
    biz_days  = df["datetime"].dt.normalize().nunique()
    expected  = biz_days * bars_per_day
    missing   = max(0, expected - total)
    miss_rate = missing / expected if expected > 0 else 0.0

    # ── 3. 가격 이상치 ─────────────────────────────────────────
    zero_price = ((df["close"] == 0) | (df["open"] == 0)).sum()

    # 봉 내 변동률: (high - low) / low
    df["_bar_range"] = (df["high"] - df["low"]) / df["low"].replace(0, np.nan)
    spikes = df[df["_bar_range"] > PRICE_SPIKE_THRESHOLD].index.tolist()

    # ── 4. OHLC 논리 검증 ──────────────────────────────────────
    ohlc_viol = (
        (df["low"]  > df["open"])  |
        (df["low"]  > df["close"]) |
        (df["high"] < df["open"])  |
        (df["high"] < df["close"])
    ).sum()

    # ── 5. 거래량 0 봉 ─────────────────────────────────────────
    zero_vol  = (df["volume"] == 0).sum()
    zvol_rate = zero_vol / total

    # ── 출력 요약 ──────────────────────────────────────────────
    date_range = (df["datetime"].min(), df["datetime"].max())
    print(f"\n{'─'*55}")
    print(f"  {label} 검증 보고서")
    print(f"{'─'*55}")
    print(f"  전체 행 수         : {total:>10,}")
    print(f"  기간               : {date_range[0].date()} ~ {date_range[1].date()}")
    print(f"  데이터 유형        : {('일봉' if is_daily else '5분봉'):>10}")
    print(f"  영업일 수          : {biz_days:>10,}")
    print(f"  예상 봉 수({bars_per_day}/일) : {expected:>10,}")
    print(f"  결측률             : {miss_rate:>9.1%}")
    print(f"  장외 봉 수         : {out_hours:>10,}")
    print(f"  가격=0 봉 수       : {zero_price:>10,}")
    print(f"  OHLC 논리 위반     : {ohlc_viol:>10,}")
    print(f"  가격 이상치 후보   : {len(spikes):>10,}건  (봉 내 변동 >{PRICE_SPIKE_THRESHOLD:.0%})")
    print(f"  거래량=0 봉 수     : {zero_vol:>10,}  ({zvol_rate:.1%})")
    print(f"{'─'*55}\n")

    # 이상치 상위 10건 출력
    if spikes:
        print("  가격 이상치 상위 10건:")
        subset = df.loc[spikes[:10], ["datetime","open","high","low","close","volume","_bar_range"]]
        print(subset.to_string(index=False))
        print()

    df.drop(columns=["_bar_range"], inplace=True)

    return {
        "total_rows":      total,
        "date_range":      date_range,
        "biz_days":        biz_days,
        "data_type":       data_type,
        "bars_per_day":    bars_per_day,
        "expected_bars":   expected,
        "missing_rate":    round(miss_rate, 4),
        "out_of_hours":    int(out_hours),
        "zero_price":      int(zero_price),
        "ohlc_violations": int(ohlc_viol),
        "price_spikes":    spikes,
        "zero_volume":     int(zero_vol),
        "zero_volume_rate": round(zvol_rate, 4),
    }


def validate_all(tickers: dict = TICKERS) -> pd.DataFrame:
    """전체 종목 검증 결과를 DataFrame으로 반환 (요약 테이블)."""
    records = []
    for ticker, name in tickers.items():
        df = load_ticker_data(ticker)
        if df.empty:
            continue
        result = validate(df, ticker)
        result["ticker"] = ticker
        result["name"]   = name
        # date_range는 문자열로 변환
        result["start"] = str(result.pop("date_range")[0].date())
        result["end"]   = str(result.pop("date_range")[1].date()) if "date_range" in result else ""
        result.pop("price_spikes", None)  # 리스트는 제외
        records.append(result)

    summary = pd.DataFrame(records)
    print("\n===== 전체 종목 검증 요약 =====")
    print(summary[["ticker","name","total_rows","missing_rate","ohlc_violations","zero_volume_rate"]].to_string(index=False))
    return summary


# ═══════════════════════════════════════════════════════════════
# 3. 샘플링 — 프로토타이핑용 Sandbox 데이터 생성
# ═══════════════════════════════════════════════════════════════

def create_sandbox(
    ticker:     str  = "005930",
    months:     int  = 3,
    save_path:  Path | None = None,
) -> pd.DataFrame:
    """
    빠른 모델 개발을 위한 소규모 sandbox 데이터셋 생성.
    지정 종목의 최근 N개월 5분봉을 /data/cleaned/sandbox_{ticker}.parquet에 저장.

    Parameters
    ----------
    ticker    : 기준 종목코드 (기본: 삼성전자 005930)
    months    : 최근 몇 개월 (기본: 3개월 ≈ 13,000봉)
    save_path : 저장 경로 (None이면 DATA_CLEAN/sandbox_{ticker}.parquet)
    """
    df = load_ticker_data(ticker)
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)

    cutoff = df["datetime"].max() - pd.DateOffset(months=months)
    sandbox = df[df["datetime"] >= cutoff].copy()
    sandbox.reset_index(drop=True, inplace=True)

    if save_path is None:
        save_path = DATA_CLEAN / f"sandbox_{ticker}.parquet"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    sandbox.to_parquet(save_path, compression="snappy", index=False)

    logger.info(
        f"[Sandbox] {ticker} 최근 {months}개월 → "
        f"{len(sandbox):,}행 저장: {save_path}"
    )
    return sandbox


# ═══════════════════════════════════════════════════════════════
# 4. 전체 검증 + 샘플링 파이프라인 (노트북에서 호출)
# ═══════════════════════════════════════════════════════════════

def run_validation_pipeline(ticker: str = "005930") -> dict:
    """
    단일 종목 검증 + sandbox 생성 전체 파이프라인.
    notebooks/01_data_collection.ipynb 에서 호출.

    Usage
    -----
    from src.data.validator import run_validation_pipeline
    report = run_validation_pipeline("005930")
    """
    df     = load_ticker_data(ticker)
    report = validate(df, ticker)
    sb     = create_sandbox(ticker, months=3)

    print(f"\n[파이프라인 완료]")
    print(f"  전체 데이터  : {len(df):,}행")
    print(f"  Sandbox 데이터: {len(sb):,}행 (최근 3개월)")
    return report
