"""
src/evaluation/backtest.py
────────────────────────────────────────────────────────────────────────────
11차시: VaR 백테스트 + 포트폴리오 시뮬레이션

■ 포함 분석
  1. 모델별 VaR 하한선 vs 실제 손실 비교
  2. 일별 초과 손실 식별 (VaR 위반 구간)
  3. 분위수별 적중률 (Quantile Hit Rate)
  4. 포트폴리오 P&L 시뮬레이션 (균등 가중)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
from scipy import stats


def quantile_hit_rate(
    y_true:    np.ndarray,
    y_pred_q:  np.ndarray,   # (n, K) — K개 분위수
    quantiles: list[float],
) -> dict[float, float]:
    """
    분위수별 적중률 = 실제값이 예측 분위수 아래에 있는 비율.
    이상적: hit_rate[τ] ≈ τ

    Returns: {tau: hit_rate}
    """
    y = np.asarray(y_true).flatten()
    result = {}
    for k, tau in enumerate(quantiles):
        hit = (y <= y_pred_q[:, k]).mean()
        result[tau] = float(hit)
    return result


def var_backtest_summary(
    y_true:    np.ndarray,
    var_lower: np.ndarray,   # τ=0.05 예측
    alpha:     float = 0.05,
) -> dict:
    """
    VaR 백테스트 종합 요약.
    """
    y    = np.asarray(y_true).flatten()
    var  = np.asarray(var_lower).flatten()
    fail = y < var

    exceedance = y[fail] - var[fail]   # 초과 손실 (음수)

    return {
        "n_total":       len(y),
        "n_failures":    int(fail.sum()),
        "failure_rate":  float(fail.mean() * 100),
        "target_rate":   float(alpha * 100),
        "max_excess":    float(exceedance.min()) if fail.any() else 0.0,
        "avg_excess":    float(exceedance.mean()) if fail.any() else 0.0,
        "failure_dates": np.where(fail)[0].tolist(),
    }


def portfolio_pnl(
    y_true:    np.ndarray,
    var_lower: np.ndarray,
    initial:   float = 1_000_000,
) -> dict:
    """
    단순 포트폴리오 P&L 시뮬레이션.
    매 시점 보유, 로그수익률 적용.
    VaR 위반 시점 표시.
    """
    y = np.asarray(y_true).flatten()
    var = np.asarray(var_lower).flatten()
    pnl = np.cumprod(np.exp(y)) * initial
    fail_mask = y < var

    return {
        "pnl":         pnl,
        "fail_mask":   fail_mask,
        "final_value": float(pnl[-1]),
        "total_return":float((pnl[-1] / initial - 1) * 100),
        "max_drawdown":float(((pnl / np.maximum.accumulate(pnl)) - 1).min() * 100),
    }
