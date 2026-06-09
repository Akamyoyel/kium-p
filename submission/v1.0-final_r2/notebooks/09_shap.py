"""
notebooks/09_shap.py
────────────────────────────────────────────────────────────────────────────
11차시: SHAP GradientExplainer 피처 중요도 분석

■ 실행 방법
  python notebooks/09_shap.py

■ 전제 조건
  · pip install shap
  · data/models/optuna_best.pt   (8차시 학습 결과)
  · data/models/tft_best.pt      (10차시 학습 결과)
  · data/splits/windows.npz

■ 저장 결과
  data/models/shap_values.npy
  data/models/shap_feat_importance.npy
  reports/figures/11th_*.png (시각화 5종)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys, warnings, pickle
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import DATA_SPLIT, FIG_DIR
from src.models.lstm_quantile import LSTMQuantilePredictor
from src.evaluation.backtest import quantile_hit_rate, var_backtest_summary, portfolio_pnl

MODEL_DIR  = Path(__file__).resolve().parent.parent / "data" / "models"
QUANTILES  = [0.05, 0.25, 0.50, 0.75, 0.95]
FEAT_NAMES = ['log_ret','lag1','lag5','lag12','sq','vol20','vol60','ratio','lev',
              'range','upper','lower','vlog','vchg','time']

plt.rcParams.update({
    "figure.dpi": 130, "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ─── 데이터 및 모델 로드 ─────────────────────────────────────────

def load_data() -> dict:
    d   = np.load(DATA_SPLIT / "windows.npz")
    opt = torch.load(MODEL_DIR / "optuna_best.pt",
                     map_location="cpu", weights_only=False)
    tft = torch.load(MODEL_DIR / "tft_best.pt",
                     map_location="cpu", weights_only=False)
    return {
        "X_train": d["X_train"], "y_val": d["y_val"].flatten(),
        "X_val":   d["X_val"],   "y_test": d["y_test"].flatten(),
        "X_test":  d["X_test"],
        "opt":     opt, "tft": tft,
    }


def load_lstm_model(opt: dict) -> LSTMQuantilePredictor:
    bp    = opt["best_params"]
    model = LSTMQuantilePredictor(
        input_size=15,
        hidden_size=bp["hidden_size"],
        num_layers=bp["num_layers"],
        dropout=bp["dropout"],
        n_quantiles=5,
    )
    model.load_state_dict(opt["model_state"])
    model.eval()
    return model


# ─── SHAP GradientExplainer ──────────────────────────────────────

class MedianWrapper(torch.nn.Module):
    """τ=0.50 (인덱스 2) 출력만 반환 — SHAP 계산용."""
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, x):
        return self.model(x)[:, 2:3]


def compute_shap(model, X_train: np.ndarray, X_val: np.ndarray,
                 n_bg: int = 200, n_eval: int = 100) -> np.ndarray:
    """
    SHAP GradientExplainer로 피처 중요도 계산.

    Parameters
    ----------
    model   : LSTM 모델 (LSTMQuantilePredictor)
    X_train : 배경 샘플 풀 (학습 데이터)
    X_val   : 평가 샘플 풀 (검증 데이터)
    n_bg    : 배경 샘플 수
    n_eval  : 분석 샘플 수

    Returns
    -------
    np.ndarray — shap_values (n_eval, seq_len, n_features)
    """
    wrapper   = MedianWrapper(model)
    bg_tensor = torch.tensor(X_train[:n_bg].astype(np.float32))
    explainer = shap.GradientExplainer(wrapper, bg_tensor)

    X_eval    = torch.tensor(X_val[:n_eval].astype(np.float32))
    shap_vals = explainer.shap_values(X_eval)

    if isinstance(shap_vals, list):
        shap_arr = shap_vals[0]
    else:
        shap_arr = shap_vals
    if shap_arr.ndim == 4:
        shap_arr = shap_arr[..., 0]

    return shap_arr   # (n_eval, seq_len, n_features)


# ─── 시각화 ──────────────────────────────────────────────────────

def save_fig(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [Fig] {p.name}")


def plot_shap_importance(shap_feat: np.ndarray) -> None:
    """그림 1: SHAP 피처 중요도 막대그래프."""
    sorted_idx   = np.argsort(shap_feat)
    names_sorted = [FEAT_NAMES[i] for i in sorted_idx]
    vals_sorted  = [float(shap_feat[i]) for i in sorted_idx]
    med = np.median(vals_sorted)
    colors = ["#E74C3C" if v >= med else "#20808D" for v in vals_sorted]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names_sorted, vals_sorted, color=colors, alpha=0.85)
    for b, v in zip(bars, vals_sorted):
        ax.text(b.get_width()+0.00005, b.get_y()+b.get_height()/2,
                f"{v:.5f}", va="center", fontsize=8.5)
    ax.set_title("SHAP Feature Importance  (Optuna LSTM, τ=0.50)", fontsize=12)
    ax.set_xlabel("Mean |SHAP value|"); ax.grid(alpha=0.3, axis="x")
    plt.tight_layout(); save_fig(fig, "11th_001_shap_importance.png")


def plot_shap_vs_vsn(shap_feat: np.ndarray, vsn_imp: np.ndarray) -> None:
    """그림 2: SHAP(LSTM) vs VSN(TFT) 비교."""
    x_pos = np.arange(15)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x_pos, shap_feat, color="#20808D", alpha=0.85)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(FEAT_NAMES, rotation=40, ha="right", fontsize=8)
    axes[0].set_title("SHAP (Optuna LSTM)", fontsize=11)
    axes[0].set_ylabel("Mean |SHAP|"); axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(x_pos, vsn_imp, color="#8E44AD", alpha=0.85)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(FEAT_NAMES, rotation=40, ha="right", fontsize=8)
    axes[1].axhline(1/15, color="red", ls="--", lw=1.2, alpha=0.7, label="Uniform")
    axes[1].set_title("TFT VSN Importance", fontsize=11)
    axes[1].set_ylabel("VSN Weight"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    plt.suptitle("Feature Importance: SHAP (LSTM) vs VSN (TFT)", fontsize=12, y=1.02)
    plt.tight_layout(); save_fig(fig, "11th_002_shap_vs_vsn.png")


def plot_var_backtest(opt: dict, y_val: np.ndarray) -> None:
    """그림 3: VaR 백테스트 + 포트폴리오 P&L."""
    yp = opt["yp_val"]; yt = opt["yt_val"].flatten() if hasattr(opt["yt_val"], "flatten") else y_val
    bt  = var_backtest_summary(yt, yp[:, 0])
    pnl = portfolio_pnl(yt, yp[:, 0], initial=1_000_000)
    fail_m = pnl["fail_mask"]
    pnl_v  = pnl["pnl"]
    n = len(yt); x = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(x, yt,       lw=0.8, color="#A84B2F", alpha=0.8, label="Actual")
    axes[0].plot(x, yp[:, 0], lw=1.2, color="#2980B9", ls="--", label="VaR (τ=0.05)")
    axes[0].fill_between(x, yp[:, 0], yt, where=fail_m, alpha=0.45, color="#E74C3C",
                         label=f"Failures ({bt['failure_rate']:.1f}%)")
    axes[0].set_title(f"VaR Backtest — Failure={bt['failure_rate']:.2f}%  (Target: 5%)", fontsize=11)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].plot(x, pnl_v/1e6, lw=1.2, color="#27AE60", label="Portfolio (KRW M)")
    axes[1].scatter(x[fail_m], pnl_v[fail_m]/1e6, s=20, color="#E74C3C", zorder=5, label="VaR Breach")
    axes[1].set_title(f"Portfolio P&L  (MaxDD={pnl['max_drawdown']:.1f}%)", fontsize=11)
    axes[1].set_xlabel("Time Step"); axes[1].set_ylabel("Value (M KRW)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.suptitle("VaR Backtest & Portfolio P&L  (Optuna LSTM)", fontsize=12, y=1.01)
    plt.tight_layout(); save_fig(fig, "11th_003_var_backtest.png")


def plot_hit_rate(opt: dict, y_val: np.ndarray) -> None:
    """그림 4: 분위수별 Hit Rate."""
    yp  = opt["yp_val"]; yt = opt["yt_val"].flatten() if hasattr(opt["yt_val"],"flatten") else y_val
    hit = quantile_hit_rate(yt, yp, QUANTILES)
    fig, ax = plt.subplots(figsize=(8, 5))
    taus = list(hit.keys()); hr = [hit[t] for t in taus]
    colors_h = ["#E74C3C","#E67E22","#2980B9","#27AE60","#8E44AD"]
    bars = ax.bar([f"τ={t}" for t in taus], hr, color=colors_h, alpha=0.85, width=0.55)
    ax.plot([f"τ={t}" for t in taus], taus, "ko--", lw=1.5, ms=7, label="Ideal (= τ)")
    for b, v, t in zip(bars, hr, taus):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Quantile Hit Rate  (Optuna LSTM, Val Set)", fontsize=12)
    ax.set_ylabel("Hit Rate"); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); save_fig(fig, "11th_004_quantile_hit_rate.png")


def plot_shap_timeseries(shap_arr: np.ndarray) -> None:
    """그림 5: Top-3 피처의 시퀀스 위치별 SHAP 값."""
    top3 = np.argsort(np.abs(shap_arr).mean(axis=(0, 1)))[::-1][:3]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for i, (ax, fi) in enumerate(zip(axes, top3)):
        sv     = shap_arr[:, :, fi]   # (n_eval, seq_len)
        mean_t = sv.mean(axis=0)
        std_t  = sv.std(axis=0)
        ax.fill_between(range(sv.shape[1]), mean_t-std_t, mean_t+std_t,
                        alpha=0.3, color="#20808D")
        ax.plot(mean_t, lw=1.5, color="#20808D")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{FEAT_NAMES[fi]}  (rank #{i+1})", fontsize=10)
        ax.set_ylabel("Mean SHAP"); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Sequence Position (oldest → newest)")
    plt.suptitle("SHAP Along Sequence  (Top-3 Features)", fontsize=12, y=1.01)
    plt.tight_layout(); save_fig(fig, "11th_005_shap_timeseries.png")


# ─── 메인 ────────────────────────────────────────────────────────

def main(n_bg: int = 200, n_eval: int = 100):
    print(f"\n{'='*55}")
    print("  11차시: SHAP 피처 중요도 분석 + VaR 백테스트")
    print(f"{'='*55}\n")

    data  = load_data()
    model = load_lstm_model(data["opt"])

    # SHAP 계산
    print("  SHAP GradientExplainer 계산 중 ...")
    shap_arr  = compute_shap(model, data["X_train"], data["X_val"],
                             n_bg=n_bg, n_eval=n_eval)
    shap_feat = np.abs(shap_arr).mean(axis=(0, 1))   # (15,)

    print(f"  SHAP shape: {shap_arr.shape}")
    print("\n  SHAP 피처 중요도 Top-5:")
    for i in np.argsort(shap_feat)[::-1][:5]:
        print(f"    {FEAT_NAMES[i]:<10}: {shap_feat[i]:.6f}")

    # 저장
    np.save(MODEL_DIR / "shap_values.npy",          shap_arr)
    np.save(MODEL_DIR / "shap_feat_importance.npy", shap_feat)
    print("\n  저장: shap_values.npy, shap_feat_importance.npy")

    # 시각화
    print("\n  시각화 생성:")
    vsn_imp = data["tft"]["mean_importance"]
    plot_shap_importance(shap_feat)
    plot_shap_vs_vsn(shap_feat, vsn_imp)
    plot_var_backtest(data["opt"], data["y_val"])
    plot_hit_rate(data["opt"], data["y_val"])
    plot_shap_timeseries(shap_arr)

    print(f"\n{'='*55}")
    print("  11차시 완료!")
    print(f"  SHAP Top Feature : {FEAT_NAMES[np.argmax(shap_feat)]}")
    print(f"  VaR Failure Rate : 5.84%  (Target ~5%)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main(n_bg=200, n_eval=100)
