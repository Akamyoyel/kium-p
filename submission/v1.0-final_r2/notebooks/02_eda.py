"""
notebooks/02_eda.py
────────────────────────────────────────────────────────────────────────────
3차시: 탐색적 데이터 분석 (EDA)
  - Jupyter 환경에서 실행하거나, python 02_eda.py 로 직접 실행 가능.
  - 실행 전 data/cleaned/sandbox_005930.parquet 또는
    data/cleaned/005930_clean.parquet 가 필요.
    (없는 경우 더미 데이터로 시연)

■ 분석 목차
  Part 1. 기초 통계 및 분포 시각화
    1-1 로그수익률 계산
    1-2 기술 통계 (평균·분산·왜도·첨도)
    1-3 히스토그램 + 정규분포 오버레이
    1-4 Q-Q Plot (정규성 검정)
    1-5 박스플롯 (시간대별 변동성)
    1-6 Rolling Volatility

  Part 2. 상관관계 분석
    2-1 Pearson 상관관계 히트맵
    2-2 ACF / PACF 분석 (수익률 & 수익률²)

  Part 3. Volatility Clustering 검증
    3-1 Ljung-Box Test (수익률², lag=10)
    3-2 ARCH-LM Test
    3-3 변동성 클러스터링 시각화
    3-4 Leverage Effect 예비 확인
────────────────────────────────────────────────────────────────────────────
"""

# ── 임포트 ───────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.stats import norm, jarque_bera, shapiro

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, FIG_DIR, TICKERS

# ── 플롯 스타일 ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       130,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",   # 서버 환경 호환
})
sns.set_theme(style="whitegrid", palette="muted")


# ═══════════════════════════════════════════════════════════════
# 0. 데이터 로드 (없으면 더미 생성)
# ═══════════════════════════════════════════════════════════════

def load_data(ticker: str = "005930") -> pd.DataFrame:
    """
    정제된 5분봉 데이터 로드.
    sandbox → cleaned → 더미 순서로 fallback.
    """
    candidates = [
        DATA_CLEAN / f"sandbox_{ticker}.parquet",
        DATA_CLEAN / f"{ticker}_clean.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            print(f"로드: {path}  ({len(df):,}행)")
            return df

    # ── 더미 데이터 (실제 수집 전 코드 테스트용) ────────────────
    print("경고: 수집된 데이터 없음 → 더미 데이터로 시연")
    np.random.seed(42)
    n = 13_000   # ≈ 삼성전자 3개월치
    dt_idx = pd.date_range("2025-01-02 09:00", periods=n, freq="5min")
    # 장외 제거 시뮬레이션
    dt_idx = dt_idx[
        (dt_idx.time >= pd.Timestamp("09:00").time()) &
        (dt_idx.time <= pd.Timestamp("15:30").time())
    ][:n]
    price = 70_000 + np.cumsum(np.random.randn(len(dt_idx)) * 150)
    vol   = (np.abs(np.random.randn(len(dt_idx))) * 10 + 5) * 1e5
    df = pd.DataFrame({
        "datetime": dt_idx,
        "open":     price * (1 + np.random.randn(len(dt_idx)) * 0.001),
        "high":     price * (1 + np.abs(np.random.randn(len(dt_idx))) * 0.002),
        "low":      price * (1 - np.abs(np.random.randn(len(dt_idx))) * 0.002),
        "close":    price,
        "volume":   vol.astype(int),
        "ticker":   ticker,
    })
    return df


# ═══════════════════════════════════════════════════════════════
# 1. 로그수익률 계산 및 기초 통계
# ═══════════════════════════════════════════════════════════════

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    로그수익률 r_t = ln(P_t / P_{t-1}) 계산.
    첫 행(NaN) 제거 후 반환.

    수학적 근거:
      로그수익률은 시계열 정상성(stationarity)에 가깝고
      연속 복리 수익률로 해석 가능하여 금융 모델링에 적합.
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df.dropna(subset=["log_ret"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def print_basic_stats(df: pd.DataFrame) -> None:
    """기술 통계 출력 (평균, 표준편차, 왜도, 초과첨도, JB 검정)."""
    r = df["log_ret"]
    jb_stat, jb_p = jarque_bera(r.dropna())

    print("\n" + "═"*52)
    print("  로그수익률 기술 통계")
    print("═"*52)
    print(f"  관측 수          : {len(r):>12,}")
    print(f"  평균 (μ)         : {r.mean():>12.6f}")
    print(f"  표준편차 (σ)     : {r.std():>12.6f}")
    print(f"  왜도 (Skewness)  : {r.skew():>12.4f}")
    print(f"  첨도 (Kurtosis)  : {r.kurtosis():>12.4f}  (초과첨도)")
    print(f"  최솟값           : {r.min():>12.6f}")
    print(f"  최댓값           : {r.max():>12.6f}")
    print(f"  Jarque-Bera 검정 : stat={jb_stat:.2f}, p={jb_p:.2e}")
    print(f"  → 정규성 {'기각 (Heavy Tail 확인)' if jb_p < 0.05 else '기각 불가':}")
    print("═"*52 + "\n")


# ═══════════════════════════════════════════════════════════════
# 2. 시각화 함수군
# ═══════════════════════════════════════════════════════════════

def plot_return_histogram(df: pd.DataFrame, ticker: str = "") -> None:
    """
    1-3. 히스토그램 + 정규분포 오버레이.

    시각화 목적:
      - 실제 분포가 정규분포 대비 첨도가 높고(뾰족)
        꼬리가 두꺼운지(Heavy Tail) 확인.
      - 분위수 경계선(5%, 95%) 표시 → VaR 영역 직관적 파악.
    """
    r = df["log_ret"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(r, bins=120, density=True, alpha=0.6, color="#20808D",
            label="실제 분포", zorder=2)

    # 정규분포 오버레이
    x = np.linspace(r.min(), r.max(), 500)
    ax.plot(x, norm.pdf(x, r.mean(), r.std()),
            "r--", lw=2, label=f"Normal(μ={r.mean():.5f}, σ={r.std():.5f})")

    # VaR 경계선
    q05 = r.quantile(0.05)
    q95 = r.quantile(0.95)
    ax.axvline(q05, color="#A84B2F", lw=1.5, ls=":", label=f"5% 분위 ({q05:.4f})")
    ax.axvline(q95, color="#1B474D", lw=1.5, ls=":", label=f"95% 분위 ({q95:.4f})")

    ax.set_title(f"{ticker} 5분봉 로그수익률 분포 (n={len(r):,})", fontsize=13)
    ax.set_xlabel("로그수익률")
    ax.set_ylabel("밀도")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _savefig(fig, f"{ticker}_hist.png")


def plot_qq(df: pd.DataFrame, ticker: str = "") -> None:
    """
    1-4. Q-Q Plot — 정규분포 이탈 시각화.

    해석:
      - 직선(45°)에서 꼬리 부분이 위아래로 벌어질수록 Heavy Tail.
      - 양 꼬리가 직선보다 위에 있으면 Leptokurtosis(첨도 과잉).
    """
    r = df["log_ret"].dropna()
    fig, ax = plt.subplots(figsize=(6, 6))
    (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm", plot=None)
    ax.scatter(osm, osr, s=6, alpha=0.4, color="#20808D", label="관측치")
    ax.plot(osm, slope * np.array(osm) + intercept,
            "r-", lw=2, label="정규분포 기댓값")
    ax.set_title(f"{ticker} Q-Q Plot", fontsize=13)
    ax.set_xlabel("이론적 분위수 (정규)")
    ax.set_ylabel("표본 분위수")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _savefig(fig, f"{ticker}_qq.png")


def plot_boxplot_by_hour(df: pd.DataFrame, ticker: str = "") -> None:
    """
    1-5. 시간대별 박스플롯 — 장 시작·마감 효과 확인.

    기대 결과:
      09:00~09:30 (장 시작)과 15:00~15:30 (마감)의
      변동성이 오후 구간 대비 크게 나타나는 'U자형' 패턴 검증.
    """
    df2 = df.copy()
    df2["hour_label"] = df2["datetime"].dt.strftime("%H:%M")

    # 30분 단위 그룹핑
    df2["30min_slot"] = (
        df2["datetime"].dt.hour * 2 +
        (df2["datetime"].dt.minute // 30)
    )
    slot_labels = (
        df2.groupby("30min_slot")["datetime"]
        .first().dt.strftime("%H:%M")
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    slots = sorted(df2["30min_slot"].unique())
    data_by_slot = [df2[df2["30min_slot"] == s]["log_ret"].dropna().values
                    for s in slots]

    ax.boxplot(data_by_slot, labels=[slot_labels.get(s, str(s)) for s in slots],
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#BCE2E7", color="#1B474D"),
               medianprops=dict(color="#A84B2F", lw=2))
    ax.set_title(f"{ticker} 시간대별 로그수익률 분포", fontsize=13)
    ax.set_xlabel("장 시간 (30분 단위)")
    ax.set_ylabel("로그수익률")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _savefig(fig, f"{ticker}_boxplot_hour.png")


def plot_rolling_volatility(df: pd.DataFrame, ticker: str = "",
                            window: int = 20) -> None:
    """
    1-6. Rolling Volatility — 변동성 집중 패턴 시각화.

    계산:
      σ_t = std(r_{t-window+1}, ..., r_t) × sqrt(78)  ← 일 변환
    (78 = 하루 5분봉 수)
    """
    df2 = df.copy().sort_values("datetime")
    df2["roll_vol"] = (
        df2["log_ret"].rolling(window).std() * np.sqrt(78) * 100  # %
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(df2["datetime"], df2["close"], color="#20808D", lw=0.8)
    axes[0].set_title(f"{ticker} 종가", fontsize=11)
    axes[0].set_ylabel("가격 (원)")

    axes[1].fill_between(df2["datetime"], df2["roll_vol"],
                         alpha=0.5, color="#A84B2F", label=f"Rolling σ ({window}봉)")
    axes[1].set_title("Rolling 변동성 (일환산 %)", fontsize=11)
    axes[1].set_ylabel("변동성 (%)")
    axes[1].legend(fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    _savefig(fig, f"{ticker}_rolling_vol.png")


# ═══════════════════════════════════════════════════════════════
# 3. 상관관계 분석
# ═══════════════════════════════════════════════════════════════

def plot_correlation_heatmap(dfs: dict) -> None:
    """
    2-1. 다종목 종가 수익률 상관관계 히트맵.

    dfs : {ticker: DataFrame}  (log_ret 컬럼 포함)

    해석 포인트:
      - 섹터 내 높은 상관 → 공통 시장 팩터 존재
      - VIX 등 외생변수와의 상관이 높을수록 외생변수 포함 타당성 증가
    """
    ret_df = pd.DataFrame({
        t: df.set_index("datetime")["log_ret"]
        for t, df in dfs.items()
        if "log_ret" in df.columns
    })
    corr = ret_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("종목별 로그수익률 Pearson 상관계수", fontsize=13)
    plt.tight_layout()
    _savefig(fig, "corr_heatmap.png")


def plot_acf_pacf(df: pd.DataFrame, ticker: str = "", lags: int = 40) -> None:
    """
    2-2. ACF / PACF — 수익률 자기상관 분석.

    ① 로그수익률 r_t ACF/PACF → 거의 없음 (약 효율적 시장 가설)
    ② 수익률 제곱 r_t² ACF/PACF → lag 1~20 강한 자기상관 (ARCH 효과)

    PACF 해석:
      r_t: 1~5 lag 유의 신호 없음 → 선형 시계열 패턴 약함
      r_t²: 1~20 lag 유의 → 분산의 시간 의존성 → 모델에 ARCH 항 필요
    """
    r  = df["log_ret"].dropna()
    r2 = r ** 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    plot_acf( r,  lags=lags, ax=axes[0, 0], title=f"{ticker} ACF (r_t)")
    plot_pacf(r,  lags=lags, ax=axes[0, 1], title=f"{ticker} PACF (r_t)",   method="ywm")
    plot_acf( r2, lags=lags, ax=axes[1, 0], title=f"{ticker} ACF (r_t²)")
    plot_pacf(r2, lags=lags, ax=axes[1, 1], title=f"{ticker} PACF (r_t²)",  method="ywm")

    for ax in axes.flat:
        ax.set_xlabel("Lag")
    plt.suptitle(f"{ticker} 로그수익률 자기상관 분석", fontsize=13, y=1.01)
    plt.tight_layout()
    _savefig(fig, f"{ticker}_acf_pacf.png")


# ═══════════════════════════════════════════════════════════════
# 4. Volatility Clustering 검증
# ═══════════════════════════════════════════════════════════════

def test_volatility_clustering(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Volatility Clustering 존재 여부 통계적 검증.

    ① Ljung-Box Test on r_t²
       H₀: r_t²에 자기상관 없음 (= 분산 독립)
       기각 → ARCH 효과 존재 (= Volatility Clustering)

    ② ARCH-LM Test (Engle's LM Test)
       H₀: No ARCH effects
       기각 → ARCH 모델 필요 확인

    ③ 결과 해석 기준
       p-value < 0.05: H₀ 기각 → Volatility Clustering 존재

    Returns
    -------
    dict  {lb_stat, lb_pval, arch_stat, arch_pval, clustering_exists}
    """
    r = df["log_ret"].dropna()

    # ── Ljung-Box Test ─────────────────────────────────────────
    lb_result = acorr_ljungbox(r ** 2, lags=[5, 10, 20], return_df=True)
    lb_stat  = lb_result["lb_stat"].values
    lb_pval  = lb_result["lb_pvalue"].values

    # ── ARCH-LM Test ───────────────────────────────────────────
    arch_lm = het_arch(r, nlags=10)
    # 반환: (LM_stat, LM_pval, F_stat, F_pval)
    arch_stat, arch_pval = arch_lm[0], arch_lm[1]

    exists = bool(lb_pval[-1] < 0.05)   # lag=20 기준

    print("\n" + "═"*55)
    print(f"  [{ticker}] Volatility Clustering 검증 결과")
    print("═"*55)
    print(f"  Ljung-Box Test (수익률²)")
    for lag, stat, pval in zip([5, 10, 20], lb_stat, lb_pval):
        sig = "*** 유의 (p<0.001)" if pval < 0.001 else ("** p<0.05" if pval < 0.05 else "기각 불가")
        print(f"    lag={lag:2d}: stat={stat:8.2f}, p={pval:.4e}  {sig}")

    print(f"\n  ARCH-LM Test (nlags=10)")
    sig = "*** 유의 (p<0.001)" if arch_pval < 0.001 else ("** p<0.05" if arch_pval < 0.05 else "기각 불가")
    print(f"    LM stat={arch_stat:.2f}, p={arch_pval:.4e}  {sig}")

    print(f"\n  최종 판정: Volatility Clustering {'존재 ✓' if exists else '미확인'}")
    print("═"*55 + "\n")

    return {
        "ticker":            ticker,
        "lb_stat_lag20":     float(lb_stat[-1]),
        "lb_pval_lag20":     float(lb_pval[-1]),
        "arch_lm_stat":      float(arch_stat),
        "arch_lm_pval":      float(arch_pval),
        "clustering_exists": exists,
    }


def plot_volatility_clustering(df: pd.DataFrame, ticker: str = "") -> None:
    """
    변동성 클러스터링 시각화 — 수익률과 수익률 절대값 time plot.

    시각적 확인 포인트:
      - 큰 수익률(절대값) 이후 큰 수익률이 연속 발생하는 군집 패턴
    """
    df2 = df.copy().sort_values("datetime")
    r   = df2["log_ret"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(df2["datetime"], r, lw=0.5, color="#20808D", alpha=0.8)
    axes[0].set_title(f"{ticker} 로그수익률 r_t", fontsize=11)
    axes[0].set_ylabel("r_t")
    axes[0].axhline(0, color="gray", lw=0.5)

    axes[1].fill_between(df2["datetime"], np.abs(r), alpha=0.6, color="#A84B2F")
    axes[1].set_title(f"|r_t| — 군집 구간에 주목", fontsize=11)
    axes[1].set_ylabel("|r_t|")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.suptitle(f"{ticker} Volatility Clustering 시각화", fontsize=13, y=1.02)
    plt.tight_layout()
    _savefig(fig, f"{ticker}_vol_clustering.png")


def plot_leverage_effect(df: pd.DataFrame, ticker: str = "",
                         lag: int = 1) -> None:
    """
    Leverage Effect 예비 확인.

    방법: 수익률 부호별 다음 봉 변동성(|r_t+lag|) 비교.
      - 음(−) 수익률 이후 변동성이 양(+) 수익률 이후보다 크면 Leverage Effect 존재.
      - 비대칭성이 있을 경우 특성공학에서 별도 피처로 반영.

    통계 검정: Welch's t-test (등분산 미가정)
    """
    df2 = df.copy().sort_values("datetime").reset_index(drop=True)
    r   = df2["log_ret"]

    # 다음 lag봉 절대 수익률
    abs_next = r.shift(-lag).abs()

    pos_vols = abs_next[r > 0].dropna()
    neg_vols = abs_next[r < 0].dropna()

    t_stat, p_val = stats.ttest_ind(neg_vols, pos_vols, equal_var=False)

    # 시각화
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.violinplot([neg_vols.values, pos_vols.values],
                  showmedians=True,
                  showextrema=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"음(−) 수익률 이후\n(n={len(neg_vols):,})",
                        f"양(+) 수익률 이후\n(n={len(pos_vols):,})"])
    ax.set_ylabel(f"|r_{{t+{lag}}}|  (다음 봉 절대 수익률)")
    ax.set_title(
        f"{ticker} Leverage Effect\n"
        f"Welch t={t_stat:.2f}, p={p_val:.4e}  "
        f"({'유의' if p_val < 0.05 else '미유의'})",
        fontsize=11
    )

    neg_mean = neg_vols.mean()
    pos_mean = pos_vols.mean()
    ratio = neg_mean / pos_mean if pos_mean > 0 else float("nan")
    print(f"\n  [{ticker}] Leverage Effect")
    print(f"    음수 수익률 이후 |r| 평균: {neg_mean:.6f}")
    print(f"    양수 수익률 이후 |r| 평균: {pos_mean:.6f}")
    print(f"    비율 (음/양):             {ratio:.3f}x")
    print(f"    Welch's t={t_stat:.2f}, p={p_val:.4e}")

    plt.tight_layout()
    _savefig(fig, f"{ticker}_leverage.png")


# ═══════════════════════════════════════════════════════════════
# 5. 결측 날짜 분포 분석
# ═══════════════════════════════════════════════════════════════

def analyze_missing_days(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    날짜별 봉 수 계산 → 결측이 많은 날짜(공휴일·데이터 오류) 식별.
    처리 방침 결정에 활용.
    """
    df2 = df.copy()
    df2["date"] = df2["datetime"].dt.normalize()
    bar_cnt = df2.groupby("date").size().rename("bar_count").reset_index()
    bar_cnt["status"] = bar_cnt["bar_count"].apply(
        lambda x: "정상(78봉)" if x >= 75
        else ("부분(30~74봉)" if x >= 30
              else "불완전(<30봉, 제거 예정)")
    )

    print(f"\n  [{ticker}] 날짜별 봉 수 분포")
    print(bar_cnt["bar_count"].describe().to_string())
    print("\n  상태별 날짜 수:")
    print(bar_cnt["status"].value_counts().to_string())

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(bar_cnt["date"], bar_cnt["bar_count"],
           width=0.8, color="#20808D", alpha=0.7)
    ax.axhline(78, color="red", lw=1, ls="--", label="정상 78봉")
    ax.axhline(30, color="orange", lw=1, ls=":",  label="최소 30봉 기준")
    ax.set_title(f"{ticker} 날짜별 5분봉 수 (결측 날짜 시각화)", fontsize=12)
    ax.set_xlabel("날짜")
    ax.set_ylabel("5분봉 수")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    _savefig(fig, f"{ticker}_bar_count.png")

    return bar_cnt


# ═══════════════════════════════════════════════════════════════
# 6. 저장 유틸 + 메인 실행
# ═══════════════════════════════════════════════════════════════

def _savefig(fig: plt.Figure, filename: str) -> None:
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=130)
    print(f"  저장: {path}")
    plt.close(fig)


def run_full_eda(ticker: str = "005930") -> None:
    """
    단일 종목 전체 EDA 실행.
    notebooks/02_eda.ipynb에서 각 셀로 분리하여 실행하거나
    scripts/  에서 python 02_eda.py 로 일괄 실행.
    """
    print(f"\n{'='*55}")
    print(f"  EDA 시작: {ticker}")
    print(f"{'='*55}")

    # 1. 로드 & 수익률 계산
    df = load_data(ticker)
    df = compute_log_returns(df)

    # 2. 기술 통계
    print_basic_stats(df)

    # 3. 시각화
    plot_return_histogram(df,        ticker=ticker)
    plot_qq(df,                       ticker=ticker)
    plot_boxplot_by_hour(df,          ticker=ticker)
    plot_rolling_volatility(df,       ticker=ticker)

    # 4. 결측 분석
    analyze_missing_days(df,          ticker=ticker)

    # 5. ACF / PACF
    plot_acf_pacf(df,                 ticker=ticker)

    # 6. Volatility Clustering
    vc_result = test_volatility_clustering(df, ticker=ticker)
    plot_volatility_clustering(df,    ticker=ticker)
    plot_leverage_effect(df,          ticker=ticker)

    print(f"\n{'='*55}")
    print(f"  EDA 완료. 저장 위치: {FIG_DIR}")
    print(f"{'='*55}\n")
    return vc_result


if __name__ == "__main__":
    run_full_eda("005930")
