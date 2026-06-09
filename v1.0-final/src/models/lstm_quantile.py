"""
src/models/lstm_quantile.py
────────────────────────────────────────────────────────────────────────────
6차시: LSTM Quantile Predictor

■ 5차시 Baseline(lstm_point.py)에서 FC 헤드만 교체
  · Single Quantile Mode (6차시): output_size=1, tau 하나
  · Multi Quantile Mode  (7차시): output_size=n_quantiles, 동시 출력

■ 구조
  Input(batch, seq, 15) → LSTM(64, 2layers) → LayerNorm → Dropout
    → FC(64 → output_size)
  output_size = 1           : 단일 τ 예측
  output_size = n_quantiles : 다중 τ 동시 예측

■ 손실 함수 연결
  · 6차시: PinballLoss(tau=0.5)  — src/losses/pinball.py
  · 7차시: MultiQuantileLoss([0.05,0.25,0.5,0.75,0.95])
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import torch
import torch.nn as nn


class LSTMQuantilePredictor(nn.Module):
    """
    LSTM 기반 분위수 예측 모델.

    Parameters
    ----------
    input_size    : 입력 피처 수 (기본 15)
    hidden_size   : LSTM 은닉 유닛 수 (기본 64)
    num_layers    : LSTM 레이어 수 (기본 2)
    dropout       : 드롭아웃 비율 (기본 0.2)
    n_quantiles   : 출력 분위수 수
                    1  → 단일 τ 모드 (6차시)
                    K  → 다중 τ 모드 (7차시, K=5)
    """

    def __init__(
        self,
        input_size:  int   = 15,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
        n_quantiles: int   = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.n_quantiles = n_quantiles

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = lstm_dropout,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        out :
          n_quantiles=1  → (batch, 1)
          n_quantiles=K  → (batch, K)
        """
        lstm_out, _ = self.lstm(x)
        last         = lstm_out[:, -1, :]   # (batch, hidden)
        out = self.norm(last)
        out = self.dropout(out)
        out = self.fc(out)                  # (batch, n_quantiles)
        return out

    def mode(self) -> str:
        return "single" if self.n_quantiles == 1 else "multi"

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"LSTMQuantilePredictor("
            f"input={self.lstm.input_size}, "
            f"hidden={self.hidden_size}, "
            f"layers={self.num_layers}, "
            f"n_quantiles={self.n_quantiles}, "
            f"params={self.count_parameters():,})"
        )
