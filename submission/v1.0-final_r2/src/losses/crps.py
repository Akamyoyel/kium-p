"""
src/losses/crps.py
────────────────────────────────────────────────────────────────────────────
9차시: CRPS (Continuous Ranked Probability Score) 구현

■ CRPS 정의
  CRPS(F, y) = ∫_{-∞}^{+∞} (F(x) - 1{x ≥ y})² dx

  여기서 F는 예측 분포(CDF), y는 실제 관측값.
  → CRPS가 작을수록 예측 분포가 실제값에 잘 맞음.

■ 분위수 기반 근사 (실용적 구현)
  분위수 예측 τ_1 < τ_2 < ... < τ_K 가 주어질 때:

    CRPS ≈ 2/K * Σ_k ρ_{τ_k}(y - ŷ_k)

  이는 K개 분위수의 Pinball Loss 평균의 2배.
  → MultiQuantileLoss의 2배와 동일한 계산.

■ 목표값
  CRPS ≤ 0.08 (프로젝트 정량 목표)

■ 참고
  Gneiting & Raftery (2007), "Strictly Proper Scoring Rules,
  Prediction, and Estimation", JASA.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════
# NumPy 기반 CRPS (평가 전용, 배열 입력)
# ═══════════════════════════════════════════════════════════════

def crps_quantile(
    y_pred: np.ndarray,    # (n, K) — K개 분위수 예측값
    y_true: np.ndarray,    # (n,)   — 실제값
    quantiles: list[float] | None = None,
) -> float:
    """
    분위수 기반 CRPS 근사.
    CRPS ≈ 2/K * Σ_k ρ_{τ_k}(y - ŷ_k)

    Parameters
    ----------
    y_pred    : (n, K) — K개 분위수 예측 배열
    y_true    : (n,)   — 실제값 배열
    quantiles : K개 분위수 수준 (기본: [0.05,0.25,0.50,0.75,0.95])

    Returns
    -------
    float — 평균 CRPS 스코어
    """
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

    y_pred = np.asarray(y_pred)   # (n, K)
    y_true = np.asarray(y_true).flatten()  # (n,)
    K = len(quantiles)

    total = 0.0
    for k, tau in enumerate(quantiles):
        e = y_true - y_pred[:, k]   # 잔차
        pinball = np.where(e >= 0, tau * e, (tau - 1.0) * e)
        total += pinball.mean()

    return float(2 * total / K)


def crps_per_sample(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    quantiles: list[float] | None = None,
) -> np.ndarray:
    """
    샘플별 CRPS 반환 (분포 분석용).

    Returns
    -------
    np.ndarray — (n,) 각 샘플의 CRPS
    """
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true).flatten()
    K = len(quantiles)

    total = np.zeros(len(y_true))
    for k, tau in enumerate(quantiles):
        e = y_true - y_pred[:, k]
        total += np.where(e >= 0, tau * e, (tau - 1.0) * e)

    return 2 * total / K


def print_crps_report(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    quantiles: list[float] | None = None,
    label: str = "Model",
) -> float:
    """CRPS 결과를 콘솔에 출력하고 평균 CRPS 반환."""
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    score = crps_quantile(y_pred, y_true, quantiles)
    per_q = []
    for k, tau in enumerate(quantiles):
        e = y_true - y_pred[:, k]
        pb = np.where(e >= 0, tau * e, (tau - 1.0) * e).mean()
        per_q.append((tau, pb))

    print(f"\n{'═'*48}")
    print(f"  CRPS Report — {label}")
    print(f"{'═'*48}")
    for tau, pb in per_q:
        print(f"  Pinball(τ={tau:.2f}) : {pb:.6f}")
    print(f"  {'─'*38}")
    print(f"  CRPS (≈ 2/K × ΣPinball) : {score:.6f}")
    target_met = "✅ 목표 달성 (≤0.08)" if score <= 0.08 else "❌ 목표 미달 (>0.08)"
    print(f"  목표 (≤ 0.08)           : {target_met}")
    print(f"{'═'*48}\n")
    return score


# ═══════════════════════════════════════════════════════════════
# PyTorch 기반 CRPS Loss (학습 최적화용)
# ═══════════════════════════════════════════════════════════════

class CRPSLoss(nn.Module):
    """
    CRPS Loss (PyTorch 텐서 기반).
    학습 시 MultiQuantileLoss 대신 직접 CRPS 최소화에 사용 가능.

    수학적으로 MultiQuantileLoss × 2 = CRPS 근사와 동치이므로,
    실제로는 MultiQuantileLoss로 최적화하고 평가만 CRPS로 수행.
    """

    def __init__(self, quantiles: list[float] | None = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        self.quantiles = quantiles
        self.register_buffer(
            "taus",
            torch.tensor(quantiles, dtype=torch.float32).unsqueeze(0),  # (1, K)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (batch, K)
        target : (batch,) 또는 (batch, 1)
        """
        y = target.squeeze(-1).unsqueeze(1)   # (batch, 1)
        e = y - pred                           # (batch, K)
        pinball = torch.where(e >= 0, self.taus * e, (self.taus - 1.0) * e)
        return 2.0 * pinball.mean()
