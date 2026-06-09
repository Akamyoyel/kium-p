"""
notebooks/03_feature_engineering.py
────────────────────────────────────────────────────────────────────────────
4차시: 데이터 전처리 및 특성공학

■ 수행 내용 (PDF 4차시 기준)
  1. Long-return(로그수익률) 계산
  2. 특성공학 15개 피처 생성
  3. MinMaxScaler 적용
  4. 시계열 Window Dataset 생성
  5. train / val / test 분할 (7:2:1)

■ 생성 피처 목록 (15개)
  [수익률 관련] 5개
    F01  log_ret          로그수익률 r_t = ln(P_t / P_{t-1})
    F02  log_ret_lag1     1봉 전 로그수익률 (r_{t-1})
    F03  log_ret_lag5     5봉 전 로그수익률 (r_{t-5})
    F04  log_ret_lag12    12봉 전 로그수익률 (1시간 전)
    F05  log_ret_squared  수익률 제곱 (r_t²) — 변동성 프록시

  [변동성 관련] 4개
    F06  roll_vol_20      20봉 Rolling 표준편차 (Volatility Clustering 캡처)
    F07  roll_vol_60      60봉 Rolling 표준편차 (장기 변동성 추세)
    F08  vol_ratio        roll_vol_20 / roll_vol_60 (단기/장기 변동성 비율)
    F09  leverage_proxy   abs(r_{t-1}) × sign(r_{t-1}) (Leverage Effect 캡처)

  [가격 구조] 3개
    F10  bar_range        (high - low) / close (봉 내 변동폭)
    F11  upper_shadow     (high - max(open,close)) / close (위꼬리)
    F12  lower_shadow     (min(open,close) - low)  / close (아래꼬리)

  [거래량 관련] 2개
    F13  vol_log          log(volume + 1) (거래량 로그 변환)
    F14  vol_change_rate  (volume - volume_lag1) / (volume_lag1 + 1) (거래량 변화율)

  [시간 피처] 1개
    F15  time_sin         sin(2π × minute_of_day / 390) (장내 시간 주기 인코딩)
────────────────────────────────────────────────────────────────────────────
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_CLEAN, DATA_FEAT, DATA_SPLIT,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    WINDOW_SIZE, HORIZON, QUANTILES, TICKERS,
)


# ═══════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════════

def load_clean_data(ticker: str = "005930") -> pd.DataFrame:
    """
    정제 완료된 데이터 로드.
    cleaner.py 실행 결과 파일 탐색 순서:
      cleaned/{ticker}_clean.parquet → cleaned/{ticker}_merged.parquet → sandbox
    """
    candidates = [
        DATA_CLEAN / f"{ticker}_clean.parquet",
        DATA_CLEAN / f"{ticker}_merged.parquet",
        DATA_CLEAN / f"sandbox_{ticker}.parquet",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"로드: {p.name}  ({len(df):,}행)")
            return df

    # 더미 데이터 (파이프라인 테스트용)
    print("경고: 정제 데이터 없음 → 더미 데이터로 파이프라인 테스트")
    return _make_dummy(ticker)


def _make_dummy(ticker: str, n: int = 15_000) -> pd.DataFrame:
    """테스트용 더미 금융 시계열 생성 (GARCH-like 변동성 집중 포함)."""
    np.random.seed(42)
    dt_idx = pd.date_range("2025-01-02 09:00", periods=n, freq="5min")
    dt_idx = dt_idx[
        (dt_idx.time >= pd.Timestamp("09:00").time()) &
        (dt_idx.time <= pd.Timestamp("15:30").time())
    ]
    n = len(dt_idx)

    # GARCH(1,1)-like 수익률 생성
    vol = np.ones(n) * 0.001
    ret = np.zeros(n)
    for i in range(1, n):
        vol[i] = np.sqrt(0.00001 + 0.1 * ret[i-1]**2 + 0.85 * vol[i-1]**2)
        ret[i] = vol[i] * np.random.randn()

    price = 70_000 * np.exp(np.cumsum(ret))
    df = pd.DataFrame({
        "datetime": dt_idx,
        "open":     price * (1 + np.random.randn(n) * 0.0005),
        "high":     price * (1 + np.abs(np.random.randn(n)) * 0.001),
        "low":      price * (1 - np.abs(np.random.randn(n)) * 0.001),
        "close":    price,
        "volume":   (np.abs(np.random.randn(n)) * 1e6 + 5e5).astype(int),
        "ticker":   ticker,
    })
    df["high"]  = df[["open","high","close"]].max(axis=1)
    df["low"]   = df[["open","low","close"]].min(axis=1)
    return df


# ═══════════════════════════════════════════════════════════════
# 2. 특성공학 — 15개 피처 생성
# ═══════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    정제된 5분봉 DataFrame → 15개 피처 추가 후 반환.
    NaN이 발생하는 초기 행은 dropna()로 제거.

    Returns
    -------
    pd.DataFrame  원본 컬럼 + 15개 피처 컬럼
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    # ── [그룹1] 수익률 관련 피처 (5개) ─────────────────────────

    # F01: 로그수익률 r_t = ln(P_t / P_{t-1})
    # 수학적 근거: 연속 복리 수익률, 덧셈 분해 가능, 정규성에 근접
    df["F01_log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # F02~F04: Lag 수익률 (시계열 자기상관 정보 제공)
    df["F02_log_ret_lag1"]  = df["F01_log_ret"].shift(1)
    df["F03_log_ret_lag5"]  = df["F01_log_ret"].shift(5)
    df["F04_log_ret_lag12"] = df["F01_log_ret"].shift(12)   # 약 1시간

    # F05: 수익률 제곱 (변동성 프록시, ARCH 효과 캡처)
    df["F05_log_ret_sq"] = df["F01_log_ret"] ** 2

    # ── [그룹2] 변동성 관련 피처 (4개) ─────────────────────────

    # F06: Rolling σ 20봉 (단기 변동성 — Volatility Clustering 핵심 피처)
    df["F06_roll_vol_20"] = df["F01_log_ret"].rolling(20).std()

    # F07: Rolling σ 60봉 (장기 변동성 추세)
    df["F07_roll_vol_60"] = df["F01_log_ret"].rolling(60).std()

    # F08: 단기/장기 변동성 비율 (1.0 초과: 단기 변동성 급등 신호)
    df["F08_vol_ratio"] = (
        df["F06_roll_vol_20"] / (df["F07_roll_vol_60"] + 1e-10)
    )

    # F09: Leverage Effect 프록시 — 음의 수익률 이후 변동성 비대칭성 캡처
    # abs(r_{t-1}) × sign(r_{t-1}) : 음수면 음수, 양수면 양수
    prev_ret = df["F01_log_ret"].shift(1)
    df["F09_leverage_proxy"] = np.abs(prev_ret) * np.sign(prev_ret)

    # ── [그룹3] 가격 구조 피처 (3개) ─────────────────────────

    # F10: 봉 내 변동폭 (고가-저가) / 종가 → 불확실성 크기
    df["F10_bar_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-10)

    # F11: 위꼬리 비율 (매도 압력 측정)
    upper_body = df[["open","close"]].max(axis=1)
    df["F11_upper_shadow"] = (df["high"] - upper_body) / (df["close"] + 1e-10)

    # F12: 아래꼬리 비율 (매수 압력 측정)
    lower_body = df[["open","close"]].min(axis=1)
    df["F12_lower_shadow"] = (lower_body - df["low"]) / (df["close"] + 1e-10)

    # ── [그룹4] 거래량 관련 피처 (2개) ─────────────────────────

    # F13: 로그 거래량 (우편향 분포 정규화)
    df["F13_vol_log"] = np.log(df["volume"] + 1)

    # F14: 거래량 변화율 (비정상적 거래량 급등 포착)
    vol_lag = df["volume"].shift(1) + 1
    df["F14_vol_change"] = (df["volume"] - df["volume"].shift(1)) / vol_lag

    # ── [그룹5] 시간 피처 (1개) ─────────────────────────────

    # F15: 장내 시간 주기 인코딩 (sin 변환)
    # 09:00=0분, 15:30=390분 → sin(2π × t / 390)
    minutes_from_open = (
        (df["datetime"].dt.hour - 9) * 60 + df["datetime"].dt.minute
    )
    df["F15_time_sin"] = np.sin(2 * np.pi * minutes_from_open / 390)

    # ── 결측 제거 (rolling window, lag 등으로 발생) ────────────
    feature_cols = [c for c in df.columns if c.startswith("F")]
    before = len(df)
    df.dropna(subset=feature_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    dropped = before - len(df)

    print(f"\n피처 생성 완료:")
    print(f"  생성 피처 수: {len(feature_cols)}개")
    print(f"  결측 제거 행: {dropped}행 (rolling window 워밍업)")
    print(f"  최종 행 수: {len(df):,}행")
    print(f"\n  피처 목록:")
    for col in feature_cols:
        desc = _FEATURE_DESC.get(col, "")
        print(f"    {col}: {desc}")

    return df


_FEATURE_DESC = {
    "F01_log_ret":        "로그수익률 r_t",
    "F02_log_ret_lag1":   "1봉 전 로그수익률",
    "F03_log_ret_lag5":   "5봉 전 로그수익률",
    "F04_log_ret_lag12":  "12봉 전 로그수익률 (~1시간)",
    "F05_log_ret_sq":     "수익률 제곱 (변동성 프록시)",
    "F06_roll_vol_20":    "Rolling σ 20봉 (단기 변동성)",
    "F07_roll_vol_60":    "Rolling σ 60봉 (장기 변동성)",
    "F08_vol_ratio":      "단기/장기 변동성 비율",
    "F09_leverage_proxy": "Leverage Effect 프록시",
    "F10_bar_range":      "봉 내 변동폭 (high-low)/close",
    "F11_upper_shadow":   "위꼬리 비율",
    "F12_lower_shadow":   "아래꼬리 비율",
    "F13_vol_log":        "로그 거래량",
    "F14_vol_change":     "거래량 변화율",
    "F15_time_sin":       "장내 시간 sin 인코딩",
}

FEATURE_COLS = [f"F{str(i).zfill(2)}_" + k for i, k in enumerate([
    "log_ret","log_ret_lag1","log_ret_lag5","log_ret_lag12","log_ret_sq",
    "roll_vol_20","roll_vol_60","vol_ratio","leverage_proxy",
    "bar_range","upper_shadow","lower_shadow",
    "vol_log","vol_change","time_sin",
], start=1)]


# ═══════════════════════════════════════════════════════════════
# 3. MinMaxScaler 적용
# ═══════════════════════════════════════════════════════════════

def scale_features(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    feature_cols: list[str],
    save_dir: Path = DATA_FEAT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Train 기준으로 MinMaxScaler 학습 후 Val / Test 변환.
    Scaler는 추론·역변환을 위해 Parquet 옆에 pickle 저장.

    ⚠ Scaler는 Train에서만 fit — Data Leakage 방지

    Returns
    -------
    (df_train_scaled, df_val_scaled, df_test_scaled, fitted_scaler)
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Train fit
    df_train = df_train.copy()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])

    # Val / Test transform
    df_val  = df_val.copy()
    df_test = df_test.copy()
    df_val[feature_cols]  = scaler.transform(df_val[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    # 저장
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_dir / "minmax_scaler.pkl")
    print(f"  Scaler 저장: {save_dir / 'minmax_scaler.pkl'}")

    return df_train, df_val, df_test, scaler


def load_scaler(save_dir: Path = DATA_FEAT) -> MinMaxScaler:
    """저장된 Scaler 로드 (추론 시 사용)."""
    path = save_dir / "minmax_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Scaler 파일 없음: {path}")
    return joblib.load(path)


# ═══════════════════════════════════════════════════════════════
# 4. train / val / test 분할 (7:2:1)
# ═══════════════════════════════════════════════════════════════

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    시계열 데이터 시간 순서 기반 분할 (shuffle 없음).
    랜덤 분할 시 미래 데이터가 학습에 사용되는 Data Leakage 발생 — 방지 필수.

    구조:
      |←───── Train (70%) ─────→|← Val (20%) →|← Test (10%) →|
      최과거                                                   최근

    Parameters
    ----------
    df          : 특성공학 완료 DataFrame (datetime 정렬 상태)
    train_ratio : Train 비율 (기본 0.7)
    val_ratio   : Validation 비율 (기본 0.2)

    Returns
    -------
    (df_train, df_val, df_test)
    """
    n = len(df)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:i_train].copy()
    df_val   = df.iloc[i_train:i_val].copy()
    df_test  = df.iloc[i_val:].copy()

    print(f"\n데이터 분할 (7:2:1):")
    print(f"  Train : {len(df_train):>7,}행  ({df_train['datetime'].min().date()} ~ {df_train['datetime'].max().date()})")
    print(f"  Val   : {len(df_val):>7,}행  ({df_val['datetime'].min().date()} ~ {df_val['datetime'].max().date()})")
    print(f"  Test  : {len(df_test):>7,}행  ({df_test['datetime'].min().date()} ~ {df_test['datetime'].max().date()})")
    return df_train, df_val, df_test


# ═══════════════════════════════════════════════════════════════
# 5. 시계열 Window Dataset 생성
# ═══════════════════════════════════════════════════════════════

def create_windows(
    df:           pd.DataFrame,
    feature_cols: list[str],
    target_col:   str   = "F01_log_ret",
    window_size:  int   = WINDOW_SIZE,
    horizon:      int   = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    슬라이딩 윈도우 방식으로 (X, y) 배열 생성.

    구조 예시 (window_size=60, horizon=1):
      X[i] = features[i : i+60]          shape: (60, n_features)
      y[i] = log_ret[i+60]               shape: (1,)

    Parameters
    ----------
    df           : 스케일된 DataFrame
    feature_cols : 입력 피처 컬럼 목록 (15개)
    target_col   : 예측 대상 컬럼 (기본: F01_log_ret)
    window_size  : 입력 시퀀스 길이 (기본: 60봉 = 5시간)
    horizon      : 예측 스텝 수 (기본: 1봉 앞)

    Returns
    -------
    X: np.ndarray  shape (n_samples, window_size, n_features)
    y: np.ndarray  shape (n_samples, horizon)
    """
    X_data = df[feature_cols].values   # (N, 15)
    y_data = df[target_col].values     # (N,)

    n = len(df)
    n_samples = n - window_size - horizon + 1

    if n_samples <= 0:
        return (
            np.zeros((0, window_size, len(feature_cols)), dtype=np.float32),
            np.zeros((0, horizon), dtype=np.float32),
        )

    X = np.zeros((n_samples, window_size, len(feature_cols)), dtype=np.float32)
    y = np.zeros((n_samples, horizon),                         dtype=np.float32)

    for i in range(n_samples):
        X[i] = X_data[i : i + window_size]
        y[i] = y_data[i + window_size : i + window_size + horizon]

    return X, y


def create_all_windows(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    feature_cols: list[str],
    save_dir: Path = DATA_SPLIT,
) -> dict:
    """
    Train / Val / Test 각각 Window 생성 후 npz 저장.

    Returns
    -------
    dict  {
        "X_train": ..., "y_train": ...,
        "X_val":   ..., "y_val":   ...,
        "X_test":  ..., "y_test":  ...,
    }
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        X, y = create_windows(df, feature_cols)
        splits[f"X_{name}"] = X
        splits[f"y_{name}"] = y
        print(f"  {name:5s}: X={X.shape}, y={y.shape}")
        if X.shape[0] == 0:
            print(
                f"    경고: {name} split 데이터가 부족합니다 "
                f"(필요 최소 행 수: window_size({WINDOW_SIZE}) + horizon({HORIZON}))"
            )

    np.savez_compressed(save_dir / "windows.npz", **splits)
    print(f"\n  Window 저장: {save_dir / 'windows.npz'}")
    return splits


def create_windows_from_scaled_splits(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    feature_cols: list[str],
    save_dir: Path = DATA_SPLIT,
    save: bool = True,
) -> dict:
    """
    경계 손실을 줄이기 위해 전체 시계열에서 윈도우를 생성한 뒤,
    타깃 시점 인덱스로 train/val/test를 분할한다.
    """
    df_all = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)
    X_all, y_all = create_windows(df_all, feature_cols)

    n_train_rows = len(df_train)
    n_val_rows   = len(df_val)
    split_val_end = n_train_rows + n_val_rows

    # horizon>1인 경우 마지막 타깃 시점을 기준으로 split 배정
    target_idx = np.arange(len(y_all)) + WINDOW_SIZE + HORIZON - 1

    train_mask = target_idx < n_train_rows
    val_mask   = (target_idx >= n_train_rows) & (target_idx < split_val_end)
    test_mask  = target_idx >= split_val_end

    splits = {
        "X_train": X_all[train_mask],
        "y_train": y_all[train_mask],
        "X_val":   X_all[val_mask],
        "y_val":   y_all[val_mask],
        "X_test":  X_all[test_mask],
        "y_test":  y_all[test_mask],
    }

    print("  Window split(타깃 시점 기준):")
    print(f"    train: X={splits['X_train'].shape}, y={splits['y_train'].shape}")
    print(f"    val  : X={splits['X_val'].shape}, y={splits['y_val'].shape}")
    print(f"    test : X={splits['X_test'].shape}, y={splits['y_test'].shape}")

    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_dir / "windows.npz", **splits)
        print(f"\n  Window 저장: {save_dir / 'windows.npz'}")

    return splits


def load_windows(save_dir: Path = DATA_SPLIT) -> dict:
    """저장된 Window npz 파일 로드."""
    path = save_dir / "windows.npz"
    if not path.exists():
        raise FileNotFoundError(f"Window 파일 없음: {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ═══════════════════════════════════════════════════════════════
# 6. 전체 파이프라인 실행
# ═══════════════════════════════════════════════════════════════

def run_feature_pipeline(ticker: str = "005930") -> dict:
    """
    4차시 전체 파이프라인:
    Load → FeatureEng → Split → Scale → Window → Save

    Usage (Jupyter)
    ---------------
    from notebooks.feature_pipeline import run_feature_pipeline
    result = run_feature_pipeline("005930")
    X_train, y_train = result["X_train"], result["y_train"]
    """
    print(f"\n{'='*55}")
    print(f"  특성공학 파이프라인: {ticker}")
    print(f"{'='*55}")

    # Step 1: 로드
    df = load_clean_data(ticker)

    # Step 2: 피처 생성
    df_feat = build_features(df)

    # Step 3: 컬럼 확인
    feat_cols = [c for c in df_feat.columns if c.startswith("F")]
    assert len(feat_cols) == 15, f"피처 수 오류: {len(feat_cols)}개 (15개 예상)"

    # Step 4: 분할 (스케일 전)
    df_train, df_val, df_test = temporal_split(df_feat)

    # Step 5: 스케일링 (train fit → val/test transform)
    print("\nMinMaxScaler 적용:")
    df_train, df_val, df_test, scaler = scale_features(
        df_train, df_val, df_test, feat_cols
    )

    # Step 6: Parquet 저장 (단일 티커 실행은 overwrite)
    for name, df_s in [("train", df_train), ("val", df_val), ("test", df_test)]:
        p = DATA_SPLIT / f"{name}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df_s.to_parquet(p, compression="snappy", index=False)
    print(f"  분할 Parquet 저장 완료: {DATA_SPLIT}")

    # Step 7: Window 생성 및 저장 (전체 시계열 윈도우 후 split)
    print("\nWindow Dataset 생성:")
    splits = create_windows_from_scaled_splits(df_train, df_val, df_test, feat_cols)

    print(f"\n{'='*55}")
    print("  파이프라인 완료!")
    print(f"  모델 입력 shape: X={splits['X_train'].shape}")
    print(f"  특성 수: {len(feat_cols)}, Window: {WINDOW_SIZE}봉, Horizon: {HORIZON}봉")
    print(f"{'='*55}\n")

    return splits


def run_feature_pipeline_all_tickers(tickers: list[str] | None = None) -> dict:
    """여러 종목을 처리해 windows.npz를 통합 데이터셋으로 생성한다."""
    if tickers is None:
        tickers = list(TICKERS.keys())

    print(f"\n{'='*55}")
    print(f"  통합 특성공학 파이프라인: {len(tickers)}개 종목")
    print(f"{'='*55}")

    merged = {
        "X_train": [], "y_train": [],
        "X_val": [],   "y_val": [],
        "X_test": [],  "y_test": [],
    }

    feat_cols_ref = None

    for ticker in tickers:
        print(f"\n[티커] {ticker}")
        df = load_clean_data(ticker)
        df_feat = build_features(df)
        feat_cols = [c for c in df_feat.columns if c.startswith("F")]
        if feat_cols_ref is None:
            feat_cols_ref = feat_cols

        df_train, df_val, df_test = temporal_split(df_feat)
        df_train, df_val, df_test, _ = scale_features(
            df_train, df_val, df_test, feat_cols
        )

        s = create_windows_from_scaled_splits(
            df_train, df_val, df_test, feat_cols, save=False
        )
        for k in merged:
            merged[k].append(s[k])

    def _stack(parts: list[np.ndarray], dims: tuple[int, ...]) -> np.ndarray:
        valid = [p for p in parts if p.size > 0]
        if valid:
            return np.concatenate(valid, axis=0)
        return np.zeros(dims, dtype=np.float32)

    n_feat = len(feat_cols_ref) if feat_cols_ref is not None else 15
    splits = {
        "X_train": _stack(merged["X_train"], (0, WINDOW_SIZE, n_feat)),
        "y_train": _stack(merged["y_train"], (0, HORIZON)),
        "X_val":   _stack(merged["X_val"],   (0, WINDOW_SIZE, n_feat)),
        "y_val":   _stack(merged["y_val"],   (0, HORIZON)),
        "X_test":  _stack(merged["X_test"],  (0, WINDOW_SIZE, n_feat)),
        "y_test":  _stack(merged["y_test"],  (0, HORIZON)),
    }

    DATA_SPLIT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_SPLIT / "windows.npz", **splits)

    print(f"\n{'='*55}")
    print("  통합 윈도우 생성 완료")
    print(f"  X_train={splits['X_train'].shape}, y_train={splits['y_train'].shape}")
    print(f"  X_val  ={splits['X_val'].shape}, y_val  ={splits['y_val'].shape}")
    print(f"  X_test ={splits['X_test'].shape}, y_test ={splits['y_test'].shape}")
    print(f"  저장: {DATA_SPLIT / 'windows.npz'}")
    print(f"{'='*55}\n")

    return splits


if __name__ == "__main__":
    result = run_feature_pipeline_all_tickers()
