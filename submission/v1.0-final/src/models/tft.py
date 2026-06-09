"""
src/models/tft.py
────────────────────────────────────────────────────────────────────────────
10차시: Temporal Fusion Transformer (TFT) — 간소화 구현

■ 원 논문
  Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting", IJF.
  → 공식 구조 대비 본 구현은 핵심 메커니즘만 추출한 교육용 경량 버전

■ 구현된 핵심 구성 요소
  1. Gated Residual Network (GRN)
     · Dense → ELU → Dense → Dropout → GLU Gate → Add & Norm
     · 선택적 정보 통과를 위한 Gating 메커니즘

  2. Variable Selection Network (VSN)
     · 입력 피처별 중요도 소프트맥스 가중합
     · 어떤 피처가 예측에 기여하는지 해석 가능

  3. Temporal Self-Attention
     · Multi-Head Attention on sequence
     · 시간적 패턴 포착 (장기 의존성)

  4. Quantile Output Head
     · FC(hidden → n_quantiles)
     · 5개 분위수 동시 출력

■ LSTM 대비 차이점
  · LSTM: 순차 처리(Sequential), 내부 gate
  · TFT:  Self-Attention, 피처 선택 가능, 해석 가능성 높음
  · TFT:  파라미터 더 많고 학습 느리지만 긴 시퀀스에 유리

■ 입력/출력
  Input : (batch, seq_len, input_size)
  Output: (batch, n_quantiles)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1. Gated Residual Network (GRN)
# ═══════════════════════════════════════════════════════════════

class GatedResidualNetwork(nn.Module):
    """
    GRN: 선택적 비선형 변환 + Gating

    structure:
      x → Dense(d) → ELU → Dense(d) → Dropout
        ↘ Dense(d, for gate)
      gate = Sigmoid(Dense(d))
      output = LayerNorm( x_proj + gate ⊙ hidden )
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.fc2     = nn.Linear(hidden_size, hidden_size)
        self.gate_fc = nn.Linear(hidden_size, hidden_size)
        self.proj    = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gate = torch.sigmoid(self.gate_fc(h))
        return self.norm(residual + gate * h)


# ═══════════════════════════════════════════════════════════════
# 2. Variable Selection Network (VSN)
# ═══════════════════════════════════════════════════════════════

class VariableSelectionNetwork(nn.Module):
    """
    VSN: 피처별 중요도 가중합

    각 피처에 GRN을 적용한 후,
    전체 입력을 기반으로 피처별 Softmax 가중치를 계산해 가중합.

    → 어떤 피처가 중요한지 weight로 해석 가능
    """
    def __init__(self, input_size: int, hidden_size: int,
                 n_features: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        # 각 피처별 GRN
        self.grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, dropout)
            for _ in range(n_features)
        ])
        # 피처 선택 weight 계산
        self.weight_grn = GatedResidualNetwork(input_size, n_features, dropout)
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq, n_features) 또는 (batch, n_features)

        Returns
        -------
        (output, weights)
          output  : (batch, seq, hidden) 또는 (batch, hidden)
          weights : (batch, seq, n_features) — 피처 중요도
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)  # (batch, 1, n_features)

        B, S, F = x.shape
        # 피처별 GRN
        processed = []
        for i, grn in enumerate(self.grns):
            xi = x[:, :, i:i+1]           # (B, S, 1)
            xi_flat = xi.reshape(B*S, 1)
            hi = grn(xi_flat).reshape(B, S, -1)
            processed.append(hi)
        processed = torch.stack(processed, dim=-1)  # (B, S, hidden, F)

        # 피처 선택 가중치
        xflat  = x.reshape(B*S, F)
        w_logit = self.weight_grn(xflat).reshape(B, S, F)  # (B, S, F)
        weights = self.softmax(w_logit)

        # 가중합
        output = (processed * weights.unsqueeze(2)).sum(dim=-1)  # (B, S, hidden)

        if squeeze:
            output  = output.squeeze(1)
            weights = weights.squeeze(1)

        return output, weights


# ═══════════════════════════════════════════════════════════════
# 3. TFT 메인 모델
# ═══════════════════════════════════════════════════════════════

class TemporalFusionTransformer(nn.Module):
    """
    경량화 TFT 구현.

    Parameters
    ----------
    input_size   : 입력 피처 수 (기본 15)
    hidden_size  : 은닉 차원 (기본 64)
    n_heads      : Multi-Head Attention 헤드 수 (기본 4)
    n_quantiles  : 출력 분위수 수 (기본 5)
    dropout      : 드롭아웃 비율 (기본 0.1)
    seq_len      : 입력 시퀀스 길이 (기본 60)
    """

    def __init__(
        self,
        input_size:  int   = 15,
        hidden_size: int   = 64,
        n_heads:     int   = 4,
        n_quantiles: int   = 5,
        dropout:     float = 0.1,
        seq_len:     int   = 60,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len     = seq_len
        self.n_quantiles = n_quantiles

        # Variable Selection
        self.vsn = VariableSelectionNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            n_features=input_size,
            dropout=dropout,
        )

        # Temporal GRN (시퀀스 전처리)
        self.temporal_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout)

        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm    = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

        # Post-Attention GRN
        self.post_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout)

        # Output Head
        self.output_norm = nn.LayerNorm(hidden_size)
        self.fc_out      = nn.Linear(hidden_size, n_quantiles)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        (out, vsn_weights)
          out         : (batch, n_quantiles)
          vsn_weights : (batch, seq_len, input_size) — 피처 중요도
        """
        B, S, _ = x.shape

        # 1. Variable Selection
        vsn_out, vsn_weights = self.vsn(x)  # (B, S, hidden)

        # 2. Temporal GRN (시퀀스 각 스텝)
        vsn_flat  = vsn_out.reshape(B * S, self.hidden_size)
        grn_out   = self.temporal_grn(vsn_flat).reshape(B, S, self.hidden_size)

        # 3. Multi-Head Self-Attention
        attn_out, _ = self.attn(grn_out, grn_out, grn_out)
        attn_out    = self.attn_norm(grn_out + self.attn_dropout(attn_out))

        # 4. Post-Attention GRN
        attn_flat = attn_out.reshape(B * S, self.hidden_size)
        post_out  = self.post_grn(attn_flat).reshape(B, S, self.hidden_size)

        # 5. 마지막 타임스텝 추출 + Output
        last   = post_out[:, -1, :]           # (B, hidden)
        last   = self.output_norm(last)
        output = self.fc_out(last)            # (B, n_quantiles)

        return output, vsn_weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"TemporalFusionTransformer("
                f"input={self.vsn.n_features}, "
                f"hidden={self.hidden_size}, "
                f"n_quantiles={self.n_quantiles}, "
                f"params={self.count_parameters():,})")
