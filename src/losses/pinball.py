"""
src/losses/pinball.py
────────────────────────────────────────────────────────────────────────────
6차시: Pinball (Quantile) Loss — PyTorch 직접 구현

■ 수학적 정의
  τ ∈ (0,1) : 분위수 수준 (예: τ=0.5 → 중앙값 회귀)

  단일 분위수:
    ρ_τ(e) = e × τ      if e ≥ 0   (과소 예측 패널티)
           = e × (τ-1)  if e < 0   (과대 예측 패널티)
    where e = y - ŷ

  다중 분위수 (Multi-Quantile):
    Loss = (1/K) × Σ_k ρ_{τ_k}(y - ŷ_k)

■ MSE와의 차이
  · MSE: 대칭 손실 → 평균 예측 (Point Prediction)
  · Pinball(τ=0.5): 대칭이나 절대값 기반 → 중앙값 예측
  · Pinball(τ<0.5): 과소 예측 페널티 ↓ → τ 분위수 예측
  · Pinball(τ>0.5): 과대 예측 페널티 ↓ → τ 분위수 예측

■ 활용
  τ=0.05 → 5%  분위수 (VaR 하한)
  τ=0.50 → 50% 분위수 (중앙값, Baseline 비교 기준)
  τ=0.95 → 95% 분위수 (VaR 상한)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """
    단일 분위수 Pinball Loss.

    Parameters
    ----------
    tau : float  분위수 수준 (0 < τ < 1)

    Usage
    -----
    criterion = PinballLoss(tau=0.5)
    loss = criterion(pred, target)  # pred, target: (batch,) 또는 (batch, 1)
    """

    def __init__(self, tau: float = 0.5):
        super().__init__()
        if not 0 < tau < 1:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.tau = tau

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (batch,) 또는 (batch, 1)
        target : pred와 동일 shape

        Returns
        -------
        loss : scalar tensor
        """
        e = target.squeeze(-1) - pred.squeeze(-1)   # 잔차 e = y - ŷ
        loss = torch.where(
            e >= 0,
            self.tau * e,           # 과소 예측: τ × |e|
            (self.tau - 1.0) * e,   # 과대 예측: (1-τ) × |e|
        )
        return loss.mean()

    def __repr__(self) -> str:
        return f"PinballLoss(tau={self.tau})"


class MultiQuantileLoss(nn.Module):
    """
    다중 분위수 Pinball Loss.
    τ 리스트 전체에 대한 평균 Pinball Loss.

    Parameters
    ----------
    quantiles : list[float]  분위수 수준 목록
                (기본: [0.05, 0.25, 0.50, 0.75, 0.95])

    Usage
    -----
    criterion = MultiQuantileLoss([0.05, 0.25, 0.5, 0.75, 0.95])
    # pred shape: (batch, n_quantiles)
    # target shape: (batch,) 또는 (batch, 1)
    loss = criterion(pred, target)
    """

    def __init__(self, quantiles: list[float] | None = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        self.quantiles = quantiles
        self.n_q = len(quantiles)
        # 각 τ를 텐서로 등록 (GPU 이동 자동 처리)
        self.register_buffer(
            "taus",
            torch.tensor(quantiles, dtype=torch.float32).unsqueeze(0),  # (1, K)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (batch, n_quantiles)  — 각 τ별 예측값
        target : (batch,) 또는 (batch, 1)

        Returns
        -------
        loss : scalar tensor (모든 분위수 평균)
        """
        y = target.squeeze(-1).unsqueeze(1)   # (batch, 1)
        e = y - pred                           # (batch, K)  잔차

        loss = torch.where(
            e >= 0,
            self.taus * e,            # 과소 예측
            (self.taus - 1.0) * e,    # 과대 예측
        )
        return loss.mean()

    def per_quantile_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[float, float]:
        """τ별 개별 Pinball Loss 반환 (평가용)."""
        y   = target.squeeze(-1).unsqueeze(1)
        e   = y - pred
        out = {}
        for i, tau in enumerate(self.quantiles):
            e_i = e[:, i]
            l_i = torch.where(e_i >= 0, tau * e_i, (tau - 1.0) * e_i)
            out[tau] = float(l_i.mean())
        return out

    def __repr__(self) -> str:
        return f"MultiQuantileLoss(quantiles={self.quantiles})"
