"""
src/evaluation/quantile_metrics.py
────────────────────────────────────────────────────────────────────────────
9차시: 분위수 예측 평가 지표

■ 포함 지표
  1. PICP  (Prediction Interval Coverage Probability)
  2. MPIW  (Mean Prediction Interval Width)
  3. Winkler Score  (커버리지 + 너비 통합 지표)
  4. VaR Coverage  (95% VaR 실제 커버리지)
  5. Kupiec Test   (VaR 모델 적절성 검정)
  6. 전체 요약 보고서

■ VaR 관련 개념
  · VaR(α): 신뢰수준 α에서 최대 예상 손실
  · α=95%: 5% 확률로 이 손실 이상 발생 가능
  · 모델 예측 τ=0.05 분위수 = 5% VaR 하한선
  · VaR Coverage = 실제로 τ=0.05 아래로 벗어난 비율
  · 이상적: VaR Coverage ≈ (1-α) = 5%

■ Kupiec Test (POF: Proportion of Failures)
  H₀: p_failure = 1 - α  (VaR 모델 올바름)
  LR = -2 ln(p₀^T1 × (1-p₀)^T0) + 2 ln(p̂^T1 × (1-p̂)^T0)
  LR ~ χ²(1)
  p-value > 0.05: H₀ 기각 불가 → VaR 모델 적절
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
from scipy import stats


# ═══════════════════════════════════════════════════════════════
# 1. PICP / MPIW / Winkler
# ═══════════════════════════════════════════════════════════════

def picp(
    y_true:   np.ndarray,
    y_lower:  np.ndarray,
    y_upper:  np.ndarray,
) -> float:
    """
    Prediction Interval Coverage Probability.
    PICP = P(y_lower ≤ y_true ≤ y_upper)
    이상적 값: 목표 신뢰수준 (예: 95% PI → PICP ≈ 0.95)
    """
    y_true  = np.asarray(y_true).flatten()
    y_lower = np.asarray(y_lower).flatten()
    y_upper = np.asarray(y_upper).flatten()
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)) * 100)


def mpiw(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Mean Prediction Interval Width. 작을수록 정밀한 예측."""
    return float(np.mean(np.asarray(y_upper) - np.asarray(y_lower)))


def winkler_score(
    y_true:   np.ndarray,
    y_lower:  np.ndarray,
    y_upper:  np.ndarray,
    alpha:    float = 0.10,   # 신뢰 수준 = 1-alpha
) -> float:
    """
    Winkler Score: 커버리지와 너비를 동시에 고려.
    W = (U-L) + (2/α)×(L-y)^+ + (2/α)×(y-U)^+
    작을수록 좋음 (구간이 좁고 커버리지가 높음).
    """
    y  = np.asarray(y_true).flatten()
    lo = np.asarray(y_lower).flatten()
    up = np.asarray(y_upper).flatten()
    width  = up - lo
    pen_lo = np.maximum(lo - y, 0) * (2 / alpha)
    pen_up = np.maximum(y - up, 0) * (2 / alpha)
    return float(np.mean(width + pen_lo + pen_up))


# ═══════════════════════════════════════════════════════════════
# 2. VaR Coverage
# ═══════════════════════════════════════════════════════════════

def var_coverage(
    y_true:    np.ndarray,
    var_lower: np.ndarray,   # τ=0.05 예측 (VaR 하한)
    alpha:     float = 0.05, # VaR 신뢰 수준 (1-α)
) -> dict:
    """
    VaR Coverage 분석.
    - failure_rate: 실제값이 VaR 하한 아래로 벗어난 비율
    - ideal: alpha (예: 5%)
    - 커버리지 = 1 - failure_rate

    Returns
    -------
    dict {failure_rate, var_coverage, ideal, over_conservative, under_conservative}
    """
    y    = np.asarray(y_true).flatten()
    var  = np.asarray(var_lower).flatten()
    fail = (y < var).mean()
    cov  = 1.0 - fail

    return {
        "failure_rate":     float(fail * 100),
        "var_coverage":     float(cov  * 100),
        "ideal_failure":    float(alpha * 100),
        "over_conservative":  fail < alpha * 0.5,   # 너무 보수적 (구간이 넓음)
        "under_conservative": fail > alpha * 2.0,   # 너무 낙관적 (구간이 좁음)
    }


# ═══════════════════════════════════════════════════════════════
# 3. Kupiec Test (Proportion of Failures Test)
# ═══════════════════════════════════════════════════════════════

def kupiec_test(
    y_true:    np.ndarray,
    var_lower: np.ndarray,
    alpha:     float = 0.05,
) -> dict:
    """
    Kupiec (1995) POF Test — VaR 모델 적절성 검정.

    H₀: 실제 실패율 = α (VaR 모델 올바름)
    LR = -2 ln L(p₀) + 2 ln L(p̂)
       ~ χ²(1)

    p-value > 0.05: H₀ 기각 불가 → VaR 모델 통계적으로 적절

    Parameters
    ----------
    alpha : 목표 VaR 신뢰 수준 (기본 0.05 = 5% VaR)
    """
    y   = np.asarray(y_true).flatten()
    var = np.asarray(var_lower).flatten()
    T   = len(y)
    T1  = int((y < var).sum())   # 실패 횟수
    T0  = T - T1                 # 성공 횟수
    p0  = alpha                  # 귀무가설 실패율
    p_hat = T1 / T               # 실제 실패율

    # LR 통계량
    if p_hat == 0:
        lr = -2 * (T0 * np.log(1 - p0))
    elif p_hat == 1:
        lr = -2 * (T1 * np.log(p0))
    else:
        lr = (-2 * (T1 * np.log(p0) + T0 * np.log(1 - p0))
              + 2 * (T1 * np.log(p_hat) + T0 * np.log(1 - p_hat)))

    p_val  = float(1 - stats.chi2.cdf(lr, df=1))
    reject = p_val < 0.05

    return {
        "T":           T,
        "T1_failures": T1,
        "T0_successes":T0,
        "p_hat":       float(p_hat * 100),
        "p0":          float(p0 * 100),
        "LR_stat":     float(lr),
        "p_value":     p_val,
        "reject_H0":   reject,
        "verdict":     "❌ VaR 모델 부적절 (과소/과대 추정)" if reject else "✅ VaR 모델 적절 (H₀ 기각 불가)",
    }


# ═══════════════════════════════════════════════════════════════
# 4. 전체 요약 보고서
# ═══════════════════════════════════════════════════════════════

def quantile_evaluation_report(
    y_pred:    np.ndarray,   # (n, K) 분위수 예측
    y_true:    np.ndarray,   # (n,)   실제값
    quantiles: list[float] | None = None,
    crps_score: float | None = None,
    label:     str = "Model",
) -> dict:
    """
    9차시 분위수 예측 전체 평가 보고서 출력.

    Returns
    -------
    dict — 전체 지표 딕셔너리
    """
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true).flatten()

    q05_idx = quantiles.index(0.05) if 0.05 in quantiles else 0
    q95_idx = quantiles.index(0.95) if 0.95 in quantiles else -1

    lower_95 = y_pred[:, q05_idx]
    upper_95 = y_pred[:, q95_idx]

    # 지표 계산
    picp_95   = picp(y_true, lower_95, upper_95)
    mpiw_95   = mpiw(lower_95, upper_95)
    wink      = winkler_score(y_true, lower_95, upper_95, alpha=0.10)
    var_res   = var_coverage(y_true, lower_95, alpha=0.05)
    kupiec    = kupiec_test(y_true, lower_95, alpha=0.05)

    print(f"\n{'═'*55}")
    print(f"  Quantile Evaluation Report — {label}")
    print(f"{'═'*55}")
    print(f"\n  [예측 구간 품질]")
    print(f"  PICP (95% PI)     : {picp_95:.2f}%  (이상: ≥90%)")
    print(f"  MPIW              : {mpiw_95:.6f}  (작을수록 정밀)")
    print(f"  Winkler Score     : {wink:.6f}  (작을수록 좋음)")
    if crps_score is not None:
        goal = "✅ 달성" if crps_score <= 0.08 else "❌ 미달"
        print(f"  CRPS              : {crps_score:.6f}  (목표≤0.08: {goal})")

    print(f"\n  [VaR(95%) 분석]")
    print(f"  실패율 (실제)     : {var_res['failure_rate']:.2f}%  (이상: 5.0%)")
    print(f"  VaR 커버리지      : {var_res['var_coverage']:.2f}%")

    print(f"\n  [Kupiec Test (VaR 적절성)]")
    print(f"  실패횟수          : {kupiec['T1_failures']} / {kupiec['T']}  ({kupiec['p_hat']:.2f}%)")
    print(f"  LR stat           : {kupiec['LR_stat']:.4f}")
    print(f"  p-value           : {kupiec['p_value']:.4f}")
    print(f"  판정              : {kupiec['verdict']}")
    print(f"\n  [목표 달성 여부]")
    picp_ok  = picp_95 >= 90.0
    crps_ok  = crps_score is not None and crps_score <= 0.08
    kupiec_ok= not kupiec["reject_H0"]
    print(f"  PICP ≥ 90%        : {'✅' if picp_ok else '❌'}  ({picp_95:.2f}%)")
    print(f"  CRPS ≤ 0.08       : {'✅' if crps_ok else '❌'}  ({crps_score:.6f})" if crps_score else "  CRPS             : 미계산")
    print(f"  Kupiec H₀ 유지    : {'✅' if kupiec_ok else '❌'}")
    print(f"{'═'*55}\n")

    return {
        "picp_95": picp_95, "mpiw_95": mpiw_95, "winkler": wink,
        "crps": crps_score, "var_coverage": var_res, "kupiec": kupiec,
    }
