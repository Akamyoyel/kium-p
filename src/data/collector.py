"""
src/data/collector.py
────────────────────────────────────────────────────────────────────────────
KIS Open API 실제 제약 기반 데이터 수집기

■ 실제 확인된 API 제약 사항
  ┌─────────────────────────────────────────────────────────────┐
  │  주식일별분봉조회 (FHKST03010230)                             │
  │  · 과거 날짜(FID_INPUT_DATE_1) 지정 조회 → 시스템상 불가      │
  │    (당일 데이터만 반환됨)                                      │
  │                                                             │
  │  주식당일분봉조회 (FHKST03010200)                             │
  │  · 당일 데이터만 제공 (전일 이전 분봉 미제공)                  │
  │  · 1회 최대 30건 → 3회 호출로 하루치(78봉) 커버               │
  └─────────────────────────────────────────────────────────────┘

■ 현재 사용 CSV 파일 정보
  · 파일명  : kospi_top10_2025.csv
  · 형태   : 일봉(Daily) OHLCV
  · 컬럼   : Date, Company, Ticker, Open, High, Low, Close, Adj Close, Volume
  · 기간   : 2025-01-02 ~ 2025-12-30
  · 종목   : KOSPI 상위 10종목 (005930.KS, 000660.KS, …)
  · 결측   : 2025-09-19 전 종목 NaN (추석 공휴일 → 제거 처리)

■ 설계 전략
  1. 과거 학습 데이터  → kospi_top10_2025.csv 로드 (load_from_csv)
  2. 당일 추가 수집    → 주식당일분봉조회 3회 호출 (collect_today)
  3. 결합 파이프라인   → load_and_update() : CSV + 당일 → Parquet 저장
────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import time
import logging
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from config.settings import DATA_RAW, DATA_CLEAN, TICKERS, API_CALL_INTERVAL
from config.kis_auth import get_access_token, get_headers

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

_BASE_URL        = os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443")
_URL_TODAY_CHART = "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
_TR_ID_TODAY     = "FHKST03010200"

# 당일 3회 호출 기준 시각 (09:30 / 12:00 / 15:30)
_SESSION_HOURS = ["093000", "120000", "153000"]


# ═══════════════════════════════════════════════════════════════
# 헬퍼: Ticker 정규화
#   005930.KS → 005930   /   005930 → 005930
# ═══════════════════════════════════════════════════════════════

def _normalize_ticker(ticker: str) -> str:
    """'.KS', '.KQ' 등 거래소 접미사 제거 → 6자리 종목코드 반환."""
    return re.sub(r'\.(KS|KQ|KX)$', '', str(ticker).strip())


# ═══════════════════════════════════════════════════════════════
# CSV 컬럼 매핑 테이블
# ═══════════════════════════════════════════════════════════════

# kospi_top10_2025.csv 헤더: Date, Company, Ticker, Open, High, Low, Close, Adj Close, Volume
_CSV_COLUMN_MAP = {
    # ── kospi_top10_2025.csv 형식 ──
    "Date":       "date",
    "Company":    "company",
    "Ticker":     "raw_ticker",
    "Open":       "open",
    "High":       "high",
    "Low":        "low",
    "Close":      "close",
    "Adj Close":  "adj_close",    # 수정 종가 (학습 시 close 대신 선택 가능)
    "Volume":     "volume",
    # ── KIS API 응답 형식 ──
    "stck_bsop_date": "date",
    "stck_cntg_hour": "time",
    "stck_oprc":      "open",
    "stck_hgpr":      "high",
    "stck_lwpr":      "low",
    "stck_prpr":      "close",
    "cntg_vol":       "volume",
    # ── 일반 소문자 형식 ──
    "datetime":   "datetime",
    "date":       "date",
    "time":       "time",
    "open":       "open",
    "high":       "high",
    "low":        "low",
    "close":      "close",
    "volume":     "volume",
}


# ═══════════════════════════════════════════════════════════════
# 1. CSV 로드 — 메인 학습 데이터
# ═══════════════════════════════════════════════════════════════

def load_from_csv(
    ticker:   str,
    csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    CSV 파일에서 특정 종목 데이터를 로드하여 정규화된 DataFrame 반환.

    Parameters
    ----------
    ticker   : 종목코드 (ex "005930" 또는 "005930.KS")
    csv_path : CSV 경로. None이면 data/raw/ 아래 자동 탐색.

    Returns
    -------
    pd.DataFrame  columns: [datetime, open, high, low, close, volume, ticker]

    탐색 순서:
      1) 인자로 지정된 csv_path
      2) data/raw/{ticker}.csv
      3) data/raw/ 내 모든 .csv 파일 (종목 컬럼 포함 형식)
      4) data/raw/{ticker}/ 폴더 내 *.csv
    """
    code = _normalize_ticker(ticker)

    # ── 탐색 경로 결정 ─────────────────────────────────────────
    if csv_path is not None:
        paths = [Path(csv_path)]
    else:
        candidates = [
            DATA_RAW / f"{code}.csv",
            DATA_RAW / "kospi_top10_2025.csv",   # 기본 제공 파일
        ]
        # 종목 폴더 내 파일들
        folder_files = sorted((DATA_RAW / code).glob("*.csv")) \
            if (DATA_RAW / code).exists() else []
        paths = [p for p in candidates if p.exists()] + folder_files

    if not paths:
        logger.error(
            f"[{code}] CSV 파일 없음.\n"
            f"  · data/raw/{code}.csv 또는\n"
            f"  · data/raw/kospi_top10_2025.csv 를 배치해 주세요."
        )
        return pd.DataFrame()

    dfs = []
    for p in paths:
        df = _read_and_filter(p, code)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        logger.warning(f"[{code}] 로드된 데이터 없음 (종목코드가 파일에 없을 수 있음)")
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    result = _deduplicate(result)
    result.sort_values("datetime", inplace=True)
    result.reset_index(drop=True, inplace=True)
    logger.info(f"[{code}] CSV 로드 완료: {len(result):,}행  ({result['datetime'].min().date()} ~ {result['datetime'].max().date()})")
    return result


def _read_and_filter(path: Path, target_code: str) -> pd.DataFrame:
    """
    단일 CSV 파일 읽기 + 컬럼 정규화 + 해당 종목 필터링.
    kospi_top10_2025.csv처럼 다종목이 섞인 파일도 처리.
    """
    try:
        raw = pd.read_csv(path, dtype=str)
    except Exception as e:
        logger.warning(f"  CSV 읽기 실패: {path} — {e}")
        return pd.DataFrame()

    # 컬럼명 정규화
    raw.rename(columns={c: _CSV_COLUMN_MAP.get(c, c) for c in raw.columns}, inplace=True)

    # ── 종목 필터링 ──────────────────────────────────────────
    # raw_ticker 컬럼이 있으면 해당 종목만 추출
    if "raw_ticker" in raw.columns:
        raw["ticker"] = raw["raw_ticker"].apply(_normalize_ticker)
        raw = raw[raw["ticker"] == target_code].copy()
        if raw.empty:
            return pd.DataFrame()
    else:
        raw["ticker"] = target_code

    # ── datetime 생성 ────────────────────────────────────────
    if "datetime" not in raw.columns:
        if "date" in raw.columns and "time" in raw.columns:
            raw["datetime"] = pd.to_datetime(
                raw["date"].str.strip() + " " + raw["time"].str.strip().str.zfill(6),
                format="%Y-%m-%d %H%M%S", errors="coerce",
            )
        elif "date" in raw.columns:
            # 일봉: Date 컬럼만 있을 때 → 당일 09:00 기준 시각 부여
            raw["datetime"] = pd.to_datetime(raw["date"].str.strip(), errors="coerce")
            # 일봉 데이터임을 플래그
            raw["_is_daily"] = True
        else:
            raw["datetime"] = pd.to_datetime(raw.iloc[:, 0], errors="coerce")

    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
    raw.dropna(subset=["datetime"], inplace=True)

    # ── 숫자 컬럼 변환 ───────────────────────────────────────
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # ── 결측치 제거 (NaN 행 — ex 추석 공휴일) ─────────────────
    before = len(raw)
    raw.dropna(subset=["open", "close", "volume"], inplace=True)
    removed = before - len(raw)
    if removed > 0:
        logger.info(f"  [{target_code}] 결측 행 제거: {removed}행 (공휴일/데이터 없음)")

    keep_cols = ["datetime", "open", "high", "low", "close", "volume", "ticker"]
    if "adj_close" in raw.columns:
        keep_cols.insert(6, "adj_close")
    if "_is_daily" in raw.columns:
        keep_cols.append("_is_daily")

    return raw[[c for c in keep_cols if c in raw.columns]].copy()


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """datetime + ticker 기준 중복 제거 (최신 행 유지)."""
    before = len(df)
    df = df.drop_duplicates(subset=["datetime", "ticker"], keep="last")
    removed = before - len(df)
    if removed:
        logger.info(f"  중복 제거: {removed}행")
    return df


# ═══════════════════════════════════════════════════════════════
# 2. 당일 5분봉 수집 (KIS API — 업데이트용)
# ═══════════════════════════════════════════════════════════════

def collect_today(
    ticker: str,
    token:  str | None = None,
) -> pd.DataFrame:
    """
    당일 5분봉 전체 수집 (3회 호출 → 78봉).

    ⚠ 제약:
      · 당일 데이터만 제공 (전일 이후 수집 불가)
      · 장 마감(15:30) 후 실행 권장

    Returns
    -------
    pd.DataFrame  정규화된 5분봉 DataFrame
    """
    code = _normalize_ticker(ticker)
    if token is None:
        try:
            token = get_access_token()
        except Exception as e:
            logger.error(f"[{code}] 토큰 발급 실패: {e}")
            return pd.DataFrame()

    all_rows = []
    for ref_hour in _SESSION_HOURS:
        rows = _fetch_today_chunk(code, ref_hour, token)
        all_rows.extend(rows)
        time.sleep(API_CALL_INTERVAL)

    if not all_rows:
        logger.warning(f"[{code}] 당일 데이터 없음 (장 마감 여부 또는 API 연결 확인)")
        return pd.DataFrame()

    df = _parse_today_output(all_rows, code)
    df = _filter_5min(df)
    df = _deduplicate(df)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"[{code}] 당일 수집 완료: {len(df)}봉")
    return df


def _fetch_today_chunk(ticker: str, hour: str, token: str) -> list[dict]:
    headers = get_headers(token)
    headers["tr_id"] = _TR_ID_TODAY
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd":         ticker,
        "fid_input_hour_1":       hour,
        "fid_pw_data_incu_yn":    "Y",
        "fid_etc_cls_code":       "",
    }
    url  = f"{_BASE_URL}{_URL_TODAY_CHART}"
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    if body.get("rt_cd") != "0":
        logger.warning(f"  [{ticker}] API 오류: {body.get('msg1')}")
        return []
    return body.get("output2", [])


def _parse_today_output(output2: list[dict], ticker: str) -> pd.DataFrame:
    records = []
    for item in output2:
        ds, ts = item.get("stck_bsop_date",""), item.get("stck_cntg_hour","")
        if len(ds) != 8 or len(ts) != 6:
            continue
        try:
            dt = datetime.strptime(ds + ts, "%Y%m%d%H%M%S")
        except ValueError:
            continue
        records.append({
            "datetime": dt,
            "open":     float(item.get("stck_oprc", 0) or 0),
            "high":     float(item.get("stck_hgpr", 0) or 0),
            "low":      float(item.get("stck_lwpr", 0) or 0),
            "close":    float(item.get("stck_prpr", 0) or 0),
            "volume":   float(item.get("cntg_vol",  0) or 0),
            "ticker":   ticker,
        })
    return pd.DataFrame(records)


def _filter_5min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["datetime"].dt.minute % 5 == 0) & (df["datetime"].dt.second == 0)
    return df[mask].copy()


# ═══════════════════════════════════════════════════════════════
# 3. CSV + 당일 결합 파이프라인
# ═══════════════════════════════════════════════════════════════

def load_and_update(
    ticker:              str,
    csv_path:            Path | str | None = None,
    collect_today_flag:  bool = False,
    save_path:           Path | None = None,
) -> pd.DataFrame:
    """
    CSV 로드 → (선택) 당일 API 수집 → 결합 → Parquet 저장.

    Parameters
    ----------
    ticker             : 종목코드 ("005930" 또는 "005930.KS")
    csv_path           : CSV 경로 (None이면 자동 탐색)
    collect_today_flag : True 시 당일 분봉 추가 수집 (장 마감 후 사용)
    save_path          : 저장 경로 (None이면 data/cleaned/{code}_merged.parquet)

    Returns
    -------
    pd.DataFrame  최종 결합 DataFrame
    """
    code = _normalize_ticker(ticker)

    # 1. CSV 로드
    df_csv = load_from_csv(code, csv_path)

    # 2. 당일 수집 (선택)
    if collect_today_flag:
        try:
            df_today = collect_today(code)
        except Exception as e:
            logger.warning(f"[{code}] 당일 수집 실패: {e}")
            df_today = pd.DataFrame()
    else:
        df_today = pd.DataFrame()

    # 3. 결합
    frames = [df for df in [df_csv, df_today] if not df.empty]
    if not frames:
        logger.error(f"[{code}] 사용 가능한 데이터 없음")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = _deduplicate(result)
    result.sort_values("datetime", inplace=True)
    result.reset_index(drop=True, inplace=True)

    # 4. 저장
    if save_path is None:
        save_path = DATA_CLEAN / f"{code}_merged.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(save_path, compression="snappy", index=False)
    logger.info(f"[{code}] 저장 완료: {len(result):,}행 → {save_path}")
    return result


def load_all_tickers(
    tickers:            dict  = TICKERS,
    csv_path:           Path | str | None = None,
    collect_today_flag: bool  = False,
) -> dict[str, pd.DataFrame]:
    """전체 종목 일괄 로드 (기본: CSV only)."""
    results = {}
    for ticker, name in tickers.items():
        logger.info(f"\n{'─'*45}\n종목: {ticker} ({name})")
        df = load_and_update(ticker, csv_path=csv_path, collect_today_flag=collect_today_flag)
        results[ticker] = df
    return results


# ═══════════════════════════════════════════════════════════════
# 4. CSV 진단 도구
# ═══════════════════════════════════════════════════════════════

def inspect_csv(csv_path: str | Path = None) -> None:
    """
    CSV 파일의 컬럼 구조를 확인하고 정규화 방법 안내.

    Usage
    -----
    from src.data.collector import inspect_csv
    inspect_csv("data/raw/kospi_top10_2025.csv")
    """
    if csv_path is None:
        csv_path = DATA_RAW / "kospi_top10_2025.csv"
    path = Path(csv_path)
    if not path.exists():
        print(f"파일 없음: {path}")
        return

    df = pd.read_csv(path, nrows=5, dtype=str)
    print(f"\n{'─'*60}")
    print(f"파일: {path.name}")
    print(f"컬럼: {list(df.columns)}")
    print(f"\n첫 3행:")
    print(df.head(3).to_string(index=False))
    print(f"{'─'*60}")
    print("\n컬럼 매핑 결과:")
    unmapped = []
    for col in df.columns:
        mapped = _CSV_COLUMN_MAP.get(col)
        if mapped:
            print(f"  ✅ '{col}' → '{mapped}'")
        else:
            print(f"  ❌ '{col}' → 미매핑 (_CSV_COLUMN_MAP에 추가 필요)")
            unmapped.append(col)
    if not unmapped:
        print("\n✅ 모든 컬럼 매핑 완료 — 바로 load_from_csv() 사용 가능합니다.")
    else:
        print(f"\n⚠ 미매핑 컬럼 {len(unmapped)}개: {unmapped}")
        print("  collector.py 상단 _CSV_COLUMN_MAP에 추가하세요.")

    # 종목 목록 출력 (다종목 파일인 경우)
    if "Ticker" in df.columns or "ticker" in df.columns:
        col = "Ticker" if "Ticker" in df.columns else "ticker"
        full = pd.read_csv(path, usecols=[col], dtype=str)
        tickers_in_file = full[col].unique()
        print(f"\n파일 내 종목 ({len(tickers_in_file)}개):")
        for t in tickers_in_file:
            code = _normalize_ticker(t)
            print(f"  {t} → 정규화: {code}")
