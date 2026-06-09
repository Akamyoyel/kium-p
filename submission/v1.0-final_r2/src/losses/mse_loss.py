"""
src/losses/mse_loss.py
────────────────────────────────────────────────────────────────────────────
5차시: MSE (Mean Squared Error) 손실 래퍼

· PyTorch nn.MSELoss의 thin wrapper
· 6차시 Quantile(Pinball) Loss와 동일한 인터페이스로 교체 가능하도록 설계
────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    MSE 손실.
    loss = mean((pred - target)^2)

    수학적 성질:
      · 대칭 손실 — 과대 예측과 과소 예측에 동일 패널티
      · 이상치에 민감 (제곱 효과)
      · Point Prediction의 표준 손실
    """

    def __init__(self):
        super().__init__()
        self._loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (batch, 1) 또는 (batch,)
        target : pred와 동일 shape
        """
        return self._loss(pred.squeeze(-1), target.squeeze(-1))
