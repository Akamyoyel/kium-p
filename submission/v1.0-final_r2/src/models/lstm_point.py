"""
src/models/lstm_point.py
────────────────────────────────────────────────────────────────────────────
5차시: LSTM Point Prediction — Baseline 모델 (MSE 손실)

■ 모델 구조
  Input  : (batch, seq_len, input_size)  — X(n, W=20, features=15)
  LSTM   : num_layers=2, hidden_size=64, dropout=0.2
  FC Head: Linear(64 → 1)
  Output : (batch, 1)  — 다음날 로그수익률 점 예측

■ 설계 원칙
  · 이 모델은 Quantile LSTM(6차시)의 Baseline 비교 대상
  · MSE 손실로 학습 → MAE, RMSE 기준 성능 측정
  · 5차시 완료 후 이 Baseline 대비 Pinball Loss 15% 감소 목표
────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn


class LSTMPointPredictor(nn.Module):
    """
    LSTM 기반 단일 점 예측(Point Prediction) 모델.

    Parameters
    ----------
    input_size   : 입력 피처 수 (기본 15)
    hidden_size  : LSTM 은닉 유닛 수 (기본 64)
    num_layers   : LSTM 레이어 수 (기본 2)
    dropout      : LSTM 드롭아웃 비율 (기본 0.2, num_layers=1 이면 자동 0)
    output_size  : 출력 크기 (기본 1 — 로그수익률 점 예측)
    """

    def __init__(
        self,
        input_size:  int   = 15,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
        output_size: int   = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # num_layers=1 이면 dropout=0 (PyTorch 경고 방지)
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = lstm_dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)    # 학습 안정화
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        out : (batch, output_size)
        """
        # LSTM — 마지막 타임스텝 hidden 사용
        lstm_out, _ = self.lstm(x)          # (batch, seq, hidden)
        last_hidden  = lstm_out[:, -1, :]   # (batch, hidden)

        out = self.norm(last_hidden)
        out = self.dropout(out)
        out = self.fc(out)                  # (batch, output_size)
        return out


# ═══════════════════════════════════════════════════════════════
# 유틸: 모델 파라미터 수 출력
# ═══════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 수 반환."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module) -> None:
    total = count_parameters(model)
    print(f"\n{'─'*45}")
    print(f"  Model: {model.__class__.__name__}")
    print(f"{'─'*45}")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:<12}: {module.__class__.__name__:<18}  ({params:,} params)")
    print(f"{'─'*45}")
    print(f"  Total trainable params: {total:,}")
    print(f"{'─'*45}\n")
