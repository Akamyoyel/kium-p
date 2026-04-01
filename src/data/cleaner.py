"""
src/data/cleaner.py
────────────────────────────────────────────────────────────────────────────
3차시 전처리: 결측치·이상치 처리 파이프라인

■ 처리 전략
  Step 1  장외 시간 봉 제거  (09:00~15:30 외)
  Step 2  가격 = 0 봉 제거   (데이터 오류)
  Step 3  OHLC 논리 위반 보정 (고 < 저 → 스왑)
  Step 4  결측 날짜 제거      (해당 날짜 전체 삭제 — 전방채움 시 수익률 왜곡)
  Step 5  이상치 Winsorizing  (로그수익률 기준 ±N·σ 초과 봉을 cap)
  Step 6  거래량 = 0 보정     (직전 봉 거래량으로 대체)
  Step 7  정렬 & 인덱스 초기화
────────────────────────────────────────────────────────────────────────────
"""

import logging
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATA_CLEAN

logger = logging.getLogger(__name__)

MARKET_START = dtime(9, 0, 0)
MARKET_END   = dtime(15, 30, 0)

WINSOR_SIGMA = 3.0   # ±3σ 초과 로그수익률 봉을 winsorize


# ═══════════════════════════════════════════════════════════════
# Step-by-Step 클리닝 함수
# ═══════════════════════════════════════════════════════════════

def _is_daily(df: pd.DataFrame) -> bool:
    """
    일봉 여부 자동 감지.
    _is_daily 플래그 컬럼이 있거나, datetime의 시/분/초가 모두 0이면 일봉으로 판단.
    """
    if "_is_daily" in df.columns and df["_is_daily"].any():
        return True
    # 시간 정보가 없는 경우 (00:00:00)
    dt = pd.to_datetime(df["datetime"])
    has_time = ((dt.dt.hour != 0) | (dt.dt.minute != 0)).any()
    return not has_time


def remove_out_of_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: 장외 시간 봉 제거 (분봉 전용 — 일봉은 스킵).
    일봉 데이터(시간 정보 없음)는 이 단계를 건너뜁니다.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 일봉이면 스킵
    if _is_daily(df):
        logger.info("  Step1 장외봉 제거: 일봉 데이터 → 스킵")
        return df

    t = df["datetime"].dt.time
    mask = (t >= MARKET_START) & (t <= MARKET_END)
    removed = (~mask).sum()
    logger.info(f"  Step1 장외봉 제거: {removed}행")
    return df[mask].copy()


def remove_zero_price(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: 가격 = 0 봉 제거 (데이터 오류)"""
    mask = (df["close"] > 0) & (df["open"] > 0) & \
           (df["high"]  > 0) & (df["low"]  > 0)
    removed = (~mask).sum()
    logger.info(f"  Step2 가격=0 제거: {removed}행")
    return df[mask].copy()


def fix_ohlc_violations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: OHLC 논리 보정
      - high < low → swap
      - open > high → open = high
      - open < low  → open = low
      - close 동일하게 보정
    """
    df = df.copy()
    # high < low 스왑
    swap = df["high"] < df["low"]
    df.loc[swap, ["high", "low"]] = df.loc[swap, ["low", "high"]].values

    # open 경계 보정
    df["open"]  = df["open"].clip(lower=df["low"], upper=df["high"])
    df["close"] = df["close"].clip(lower=df["low"], upper=df["high"])

    logger.info(f"  Step3 OHLC 보정: {swap.sum()}행 스왑")
    return df


def remove_incomplete_days(df: pd.DataFrame, min_bars: int = None) -> pd.DataFrame:
    """
    Step 4: 봉 수가 min_bars 미만인 날짜 전체 제거.
    공휴일·조기 폐장일·데이터 수집 실패일을 제거.
    (전방채움 대신 삭제 → 수익률 왜곡 방지)
    """
    df = df.copy()
    if min_bars is None:
        min_bars = 1 if _is_daily(df) else 30
    df["_date"] = df["datetime"].dt.normalize()
    bar_counts  = df.groupby("_date").size()
    valid_dates = bar_counts[bar_counts >= min_bars].index
    before = len(df)
    df = df[df["_date"].isin(valid_dates)].copy()
    df.drop(columns=["_date"], inplace=True)
    logger.info(f"  Step4 불완전 날짜 제거: {before - len(df)}행")
    return df


def winsorize_returns(df: pd.DataFrame, sigma: float = WINSOR_SIGMA) -> pd.DataFrame:
    """
    Step 5: 종가 기준 로그수익률 Winsorizing.
    ±sigma·σ 초과 봉의 close를 상/하한으로 cap.
    (open, high, low도 비율에 맞게 조정)

    ※ 서킷브레이커 등 실제 이벤트는 보존:
       이상치 flag만 기록하고, cap은 ±sigma·σ 기준 적용
    """
    df = df.copy()
    df.sort_values("datetime", inplace=True)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    mu    = df["log_ret"].mean()
    sigma_val = df["log_ret"].std()
    upper = mu + sigma * sigma_val
    lower = mu - sigma * sigma_val

    # 이상치 플래그
    df["is_spike"] = (df["log_ret"] > upper) | (df["log_ret"] < lower)
    spike_cnt = df["is_spike"].sum()

    # cap 적용 (close 기준, ratio로 나머지 가격 조정)
    cap_upper = df["close"].shift(1) * np.exp(upper)
    cap_lower = df["close"].shift(1) * np.exp(lower)

    spike_up   = df["log_ret"] > upper
    spike_down = df["log_ret"] < lower

    # 비율 보정
    for col in ["open", "high", "low"]:
        ratio = df[col] / df["close"]
        df.loc[spike_up,   "close"] = cap_upper[spike_up]
        df.loc[spike_down, "close"] = cap_lower[spike_down]
        df[col] = df["close"] * ratio

    # OHLC 재정렬
    df["high"] = df[["open","high","low","close"]].max(axis=1)
    df["low"]  = df[["open","high","low","close"]].min(axis=1)

    logger.info(f"  Step5 Winsorizing ±{sigma}σ: {spike_cnt}봉 조정")
    df.drop(columns=["log_ret"], inplace=True)
    return df


def fix_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: 거래량 = 0 봉 → 직전 봉 거래량으로 대체 (forward fill)"""
    df = df.copy()
    zero_cnt = (df["volume"] == 0).sum()
    df.loc[df["volume"] == 0, "volume"] = np.nan
    df["volume"] = df["volume"].ffill()
    logger.info(f"  Step6 거래량=0 보정: {zero_cnt}봉")
    return df


# ═══════════════════════════════════════════════════════════════
# 통합 파이프라인
# ═══════════════════════════════════════════════════════════════

def clean(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    전체 클리닝 파이프라인 실행 후 정렬된 DataFrame 반환.

    Usage
    -----
    from src.data.cleaner import clean
    df_clean = clean(df_raw, ticker="005930")
    """
    label = f"[{ticker}]" if ticker else ""
    logger.info(f"\n{label} 클리닝 시작 (원본 {len(df):,}행)")

    df = remove_out_of_hours(df)
    df = remove_zero_price(df)
    df = fix_ohlc_violations(df)
    df = remove_incomplete_days(df)
    df = winsorize_returns(df)
    df = fix_zero_volume(df)

    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"{label} 클리닝 완료 → {len(df):,}행\n")
    return df


def clean_and_save(
    df:       pd.DataFrame,
    ticker:   str,
    save_dir: Path = DATA_CLEAN,
) -> pd.DataFrame:
    """클리닝 후 cleaned/{ticker}_clean.parquet로 저장."""
    df_clean = clean(df, ticker)
    path = save_dir / f"{ticker}_clean.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(path, compression="snappy", index=False)
    logger.info(f"  저장 완료: {path}")
    return df_clean
