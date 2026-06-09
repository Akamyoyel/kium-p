"""
notebooks/05_quantile_lstm.py
────────────────────────────────────────────────────────────────────────────
6차시: Single Quantile LSTM (τ=0.5) — Pinball Loss
7차시: Multi-Quantile LSTM ([0.05,0.25,0.50,0.75,0.95]) — MultiQuantileLoss

■ 6차시 목표
  · Pinball Loss 직접 구현 및 검증
  · τ=0.5 예측 → Baseline(MSE) MAE 대비 15% 이상 개선 목표
  · Baseline MAE=0.2103 → 목표: < 0.1788

■ 7차시 목표
  · 5개 분위수 동시 예측 (FC Head: 64→5)
  · 95% 예측 구간 시각화 (Prediction Interval)
  · PICP(Prediction Interval Coverage Probability) ≥ 90% 목표
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config.settings import DATA_SPLIT, FIG_DIR
from src.models.lstm_quantile import LSTMQuantilePredictor
from src.losses.pinball import PinballLoss, MultiQuantileLoss
from src.evaluation.metrics import regression_report, mae, rmse, directional_accuracy

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "font.family": "DejaVu Sans",
    "axes.unicode_minus": False, "axes.spines.top": False,
    "axes.spines.right": False,
})

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
COLORS    = {
    0.05: "#E74C3C", 0.25: "#E67E22",
    0.50: "#2980B9", 0.75: "#27AE60", 0.95: "#8E44AD",
}

# ═══════════════════════════════════════════════════════════════
# 공통 유틸
# ═══════════════════════════════════════════════════════════════

def load_windows() -> dict:
    d = np.load(DATA_SPLIT / "windows.npz")
    return {k: torch.tensor(d[k]) for k in d.files}


def get_loaders(splits: dict, batch_size: int = 128):
    def make(X, y, shuffle):
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
    return (make(splits["X_train"], splits["y_train"], True),
            make(splits["X_val"],   splits["y_val"],   False),
            make(splits["X_test"],  splits["y_test"],  False))


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train(); total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval(); total = 0.0; ps, ys = [], []
    n_samples = len(loader.dataset)
    if n_samples == 0:
        return np.nan, np.empty((0, model.n_quantiles)), np.empty((0,))
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total += criterion(pred, y).item() * X.size(0)
        ps.append(pred.cpu().numpy()); ys.append(y.cpu().numpy())
    return total / n_samples, np.concatenate(ps), np.concatenate(ys)


def run_training(model, criterion, splits, device,
                 epochs=100, batch_size=128, patience=15, lr=1e-3, label=""):
    tr_ld, vl_ld, te_ld = get_loaders(splits, batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=7)
    th, vh = [], []; best_vl = 1e9; no_imp = 0; best_st = None

    for ep in range(1, epochs + 1):
        tl = train_epoch(model, tr_ld, opt, criterion, device)
        vl, _, _ = eval_epoch(model, vl_ld, criterion, device)
        sch.step(vl); th.append(tl); vh.append(vl)
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep % 20 == 0 or ep == 1:
            print(f"  [{label}] Ep{ep:3d}  tr={tl:.5f}  vl={vl:.5f}  best={best_vl:.5f}")
        if no_imp >= patience:
            print(f"  ⏹ Early stop @ ep{ep}")
            break

    model.load_state_dict(best_st)
    _, yp_v, yt_v = eval_epoch(model, vl_ld, criterion, device)
    _, yp_t, yt_t = eval_epoch(model, te_ld, criterion, device)
    return {"label": label, "train_hist": np.array(th), "val_hist": np.array(vh),
            "best_val_loss": best_vl, "yp_val": yp_v, "yt_val": yt_v,
            "yp_test": yp_t, "yt_test": yt_t, "model_state": best_st}


# ═══════════════════════════════════════════════════════════════
# 6차시: Single Quantile τ=0.5
# ═══════════════════════════════════════════════════════════════

def run_week6(splits: dict, device: torch.device) -> dict:
    """τ=0.5 단일 Quantile LSTM 학습."""
    print(f"\n{'='*55}")
    print("  6차시: Single Quantile LSTM  (τ=0.5)")
    print(f"{'='*55}")

    model = LSTMQuantilePredictor(input_size=15, hidden_size=64,
                                   num_layers=2, n_quantiles=1)
    print(f"  모델: {model}")
    criterion = PinballLoss(tau=0.5)

    res = run_training(model, criterion, splits, device,
                       epochs=100, batch_size=128, patience=15,
                       lr=1e-3, label="Q_τ=0.5")

    # 성능 보고 (MAE 기준 Baseline 비교)
    print("\n  [6차시 성능 보고]")
    vm = regression_report(res["yt_val"].flatten(),
                           res["yp_val"].flatten(), "Week6 τ=0.5  Val")
    tm = regression_report(res["yt_test"].flatten(),
                           res["yp_test"].flatten(), "Week6 τ=0.5  Test")

    baseline_mae = 0.210272
    improvement  = (baseline_mae - vm["mae"]) / baseline_mae * 100
    print(f"\n  Baseline MAE  : {baseline_mae:.6f}")
    print(f"  Quantile MAE  : {vm['mae']:.6f}")
    print(f"  개선율         : {improvement:+.2f}%  (목표: > +15%)")

    res["val_metrics"]  = vm
    res["test_metrics"] = tm
    res["improvement"]  = improvement

    torch.save({"model_state": res["model_state"], "val_metrics": vm,
                "test_metrics": tm, "improvement": improvement},
               MODEL_DIR / "lstm_q50_best.pt")
    print("  저장: data/models/lstm_q50_best.pt")
    return res


# ═══════════════════════════════════════════════════════════════
# 7차시: Multi-Quantile [0.05, 0.25, 0.50, 0.75, 0.95]
# ═══════════════════════════════════════════════════════════════

def run_week7(splits: dict, device: torch.device) -> dict:
    """5개 분위수 동시 예측 Multi-Quantile LSTM."""
    print(f"\n{'='*55}")
    print("  7차시: Multi-Quantile LSTM  (τ = [0.05,0.25,0.50,0.75,0.95])")
    print(f"{'='*55}")

    model = LSTMQuantilePredictor(input_size=15, hidden_size=64,
                                   num_layers=2, n_quantiles=5)
    print(f"  모델: {model}")
    criterion = MultiQuantileLoss(QUANTILES)

    res = run_training(model, criterion, splits, device,
                       epochs=100, batch_size=128, patience=15,
                       lr=1e-3, label="Multi-Q")

    # τ별 개별 Pinball Loss
    X_v = splits["X_val"].to(device)
    y_v = splits["y_val"].to(device)
    model.eval()
    with torch.no_grad():
        pred_v = model(X_v)   # (754, 5)
    pq = criterion.per_quantile_loss(pred_v, y_v)
    print("\n  τ별 Pinball Loss (Val):")
    for tau, loss_v in pq.items():
        print(f"    τ={tau:.2f}: {loss_v:.6f}")

    # PICP: 95% 예측구간 실제 커버리지
    yp_v = res["yp_val"]    # (754, 5)
    yt_v = res["yt_val"].flatten()
    lower, upper = yp_v[:, 0], yp_v[:, 4]   # τ=0.05, τ=0.95
    picp_95 = float(np.mean((yt_v >= lower) & (yt_v <= upper)) * 100)
    print(f"\n  PICP (95% 예측구간): {picp_95:.2f}%  (목표: ≥ 90%)")

    # MPIW: 평균 구간 너비
    mpiw = float(np.mean(upper - lower))
    print(f"  MPIW (평균 구간 너비): {mpiw:.6f}")

    res["pq_losses"] = pq
    res["picp_95"]   = picp_95
    res["mpiw"]      = mpiw

    torch.save({"model_state": res["model_state"], "quantiles": QUANTILES,
                "pq_losses": pq, "picp_95": picp_95, "mpiw": mpiw},
               MODEL_DIR / "lstm_multiq_best.pt")
    print("  저장: data/models/lstm_multiq_best.pt")
    return res


# ═══════════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════════

def save_fig(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] 저장: {p.name}")


# ── 6차시 시각화 ──────────────────────────────────────────────

def plot_w6_training(r6: dict) -> None:
    """그림 1: 6차시 학습 곡선."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in zip(axes,
        ["train_hist", "val_hist"],
        ["Train Pinball Loss (τ=0.5)", "Val Pinball Loss (τ=0.5)"]):
        ax.plot(r6[key], lw=1.5, color="#2980B9")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Pinball Loss")
        ax.grid(alpha=0.3)
    best_ep = int(np.argmin(r6["val_hist"]))
    axes[1].axvline(best_ep, color="red", ls=":", lw=1.5)
    axes[1].annotate(f"Best\nEp={best_ep+1}",
        xy=(best_ep, r6["val_hist"][best_ep]),
        xytext=(best_ep+3, r6["val_hist"][best_ep]+0.001),
        fontsize=8, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1))
    plt.tight_layout()
    save_fig(fig, "6th_001_training_curve.png")


def plot_w6_pred(r6: dict) -> None:
    """그림 2: 6차시 예측 vs 실제."""
    yp = r6["yp_val"].flatten()
    yt = r6["yt_val"].flatten()
    n  = len(yt)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(range(n), yt, lw=0.8, color="#20808D", label="Actual", alpha=0.9)
    axes[0].plot(range(n), yp, lw=0.8, color="#A84B2F", ls="--",
                 label="Predicted (τ=0.5)", alpha=0.85)
    axes[0].set_title("τ=0.5 Predicted vs Actual  (Val Set)", fontsize=11)
    axes[0].set_xlabel("Time Step"); axes[0].set_ylabel("Scaled Log-Return")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    vm = r6["val_metrics"]
    txt = (f"MAE  = {vm['mae']:.5f}\nRMSE = {vm['rmse']:.5f}\n"
           f"R²   = {vm['r2']:.5f}\nDirAcc= {vm['dir_acc']:.1f}%\n"
           f"Improve= {r6['improvement']:+.2f}%")
    axes[0].text(0.02, 0.97, txt, transform=axes[0].transAxes,
        fontsize=7.5, va="top", fontfamily="monospace",
        bbox=dict(fc="white", ec="#ccc", alpha=0.9, boxstyle="round,pad=0.3"))
    lim = max(abs(yt).max(), abs(yp).max()) * 1.1
    axes[1].scatter(yt, yp, s=8, alpha=0.4, color="#2980B9")
    axes[1].plot([-lim,lim],[-lim,lim],"r--",lw=1.5,label="Perfect Fit")
    axes[1].set_xlim(-lim,lim); axes[1].set_ylim(-lim,lim)
    axes[1].set_title("Scatter: τ=0.5 Predicted vs Actual", fontsize=11)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted (τ=0.5)")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "6th_002_pred_vs_true.png")


def plot_w6_loss_compare(r6: dict) -> None:
    """그림 3: τ=0.5 손실 vs MSE 비교 개념도 + Pinball 비대칭 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    e = np.linspace(-0.15, 0.15, 400)
    tau = 0.5
    pinball = np.where(e >= 0, tau*e, (tau-1)*e)
    mse     = e**2
    axes[0].plot(e, pinball, lw=2, color="#2980B9", label="Pinball(τ=0.5)")
    axes[0].plot(e, mse/max(mse)*max(pinball)*0.85, lw=2, color="#A84B2F",
                 ls="--", label="MSE (scaled for comparison)")
    axes[0].axvline(0, color="gray", lw=0.8)
    axes[0].set_title("Loss Function Comparison\nPinball(τ=0.5) vs MSE", fontsize=11)
    axes[0].set_xlabel("Residual  e = y - ŷ")
    axes[0].set_ylabel("Loss Value"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    for tau_i, c in [(0.05,"#E74C3C"),(0.25,"#E67E22"),(0.50,"#2980B9"),
                     (0.75,"#27AE60"),(0.95,"#8E44AD")]:
        p_i = np.where(e>=0, tau_i*e, (tau_i-1)*e)
        axes[1].plot(e, p_i, lw=1.8, color=c, label=f"τ={tau_i}")
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[1].set_title("Pinball Loss Shape by τ\n(Asymmetric Penalty)", fontsize=11)
    axes[1].set_xlabel("Residual  e = y - ŷ"); axes[1].set_ylabel("Loss Value")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "6th_003_pinball_shape.png")


def plot_w6_baseline_compare(r6: dict) -> None:
    """그림 4: Baseline(MSE) vs Quantile(τ=0.5) 성능 비교."""
    baseline = {"MAE": 0.210272, "RMSE": 0.264297, "DirAcc%": 71.09}
    quantile = {"MAE": r6["val_metrics"]["mae"],
                "RMSE": r6["val_metrics"]["rmse"],
                "DirAcc%": r6["val_metrics"]["dir_acc"]}
    metrics  = list(baseline.keys())
    b_vals   = [baseline[m] for m in metrics]
    q_vals   = [quantile[m] for m in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.35
    b1 = ax.bar(x-w/2, b_vals, w, label="Baseline (MSE)",   color="#A84B2F", alpha=0.8)
    b2 = ax.bar(x+w/2, q_vals, w, label="Quantile (τ=0.5)", color="#2980B9", alpha=0.8)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.015,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title("Baseline (MSE) vs Quantile τ=0.5  —  Val Set", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis="y")
    imp = r6["improvement"]
    ax.text(0.5, 0.95, f"MAE improvement: {imp:+.2f}%",
        transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold",
        color="#27AE60" if imp > 0 else "#E74C3C",
        bbox=dict(fc="white", ec="#ccc", boxstyle="round,pad=0.3"))
    plt.tight_layout()
    save_fig(fig, "6th_004_baseline_compare.png")


def plot_w6_residual(r6: dict) -> None:
    """그림 5: 6차시 잔차 분석."""
    from scipy.stats import norm
    yp = r6["yp_val"].flatten(); yt = r6["yt_val"].flatten()
    resid = yt - yp; mu, sig = resid.mean(), resid.std()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(resid, bins=40, density=True, alpha=0.7, color="#2980B9")
    xr = np.linspace(resid.min(), resid.max(), 300)
    axes[0].plot(xr, norm.pdf(xr, mu, sig), "r--", lw=2,
                 label=f"Normal(μ={mu:.4f}, σ={sig:.4f})")
    axes[0].set_title("Residual Distribution  (τ=0.5, Val)", fontsize=11)
    axes[0].set_xlabel("Residual"); axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(resid, lw=0.7, color="#A84B2F", alpha=0.8)
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].axhline( 2*sig, color="#FFC553", lw=1.2, ls="--", label=f"±2σ")
    axes[1].axhline(-2*sig, color="#FFC553", lw=1.2, ls="--")
    axes[1].set_title("Residual Time Series  (τ=0.5)", fontsize=11)
    axes[1].set_xlabel("Time Step"); axes[1].set_ylabel("Residual")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "6th_005_residual.png")


# ── 7차시 시각화 ──────────────────────────────────────────────

def plot_w7_training(r7: dict) -> None:
    """그림 6: 7차시 학습 곡선."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r7["train_hist"], lw=1.5, color="#8E44AD", label="Train")
    ax.plot(r7["val_hist"],   lw=1.5, color="#E67E22", label="Val")
    best_ep = int(np.argmin(r7["val_hist"]))
    ax.axvline(best_ep, color="red", ls=":", lw=1.5)
    ax.set_title("Multi-Quantile LSTM — Train/Val MultiQuantileLoss", fontsize=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MultiQuantile Loss")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "7th_001_training_curve.png")


def plot_w7_prediction_interval(r7: dict) -> None:
    """그림 7: 예측 구간 시각화 (핵심 — 7차시 대표 그림)."""
    yp = r7["yp_val"]    # (n, 5)
    yt = r7["yt_val"].flatten()
    n  = min(200, len(yt))     # 앞 200개만 표시

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n)
    # 95% 구간
    ax.fill_between(x, yp[:n,0], yp[:n,4], alpha=0.18, color="#8E44AD",
                    label="95% PI  (τ=0.05~0.95)")
    # 50% 구간
    ax.fill_between(x, yp[:n,1], yp[:n,3], alpha=0.30, color="#2980B9",
                    label="50% PI  (τ=0.25~0.75)")
    # 중앙값
    ax.plot(x, yp[:n,2], lw=1.2, color="#2980B9", label="Median (τ=0.50)")
    # 실제값
    ax.plot(x, yt[:n], lw=0.9, color="#A84B2F", alpha=0.9, label="Actual")
    ax.set_title("Multi-Quantile LSTM — Prediction Interval (Val Set, first 200 steps)",
                 fontsize=12)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Scaled Log-Return")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
    picp = r7["picp_95"]
    ax.text(0.02, 0.97,
        f"PICP(95% PI) = {picp:.1f}%\nMPIW = {r7['mpiw']:.5f}",
        transform=ax.transAxes, fontsize=9, va="top", fontfamily="monospace",
        bbox=dict(fc="white", ec="#ccc", alpha=0.9, boxstyle="round,pad=0.3"))
    plt.tight_layout()
    save_fig(fig, "7th_002_prediction_interval.png")


def plot_w7_quantile_each(r7: dict) -> None:
    """그림 8: τ별 개별 Pinball Loss 막대그래프."""
    pq = r7["pq_losses"]
    taus = list(pq.keys()); vals = list(pq.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(t) for t in taus], vals,
                  color=[COLORS[t] for t in taus], alpha=0.85, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.015,
                f"{v:.5f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Per-Quantile Pinball Loss  (Val Set)", fontsize=12)
    ax.set_xlabel("τ (Quantile Level)"); ax.set_ylabel("Pinball Loss")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    save_fig(fig, "7th_003_per_quantile_loss.png")


def plot_w7_coverage(r7: dict) -> None:
    """그림 9: 다양한 구간 폭의 실제 커버리지 분석 (Coverage-Width Trade-off)."""
    yp = r7["yp_val"]    # (n,5)
    yt = r7["yt_val"].flatten()
    # 구간: [q05,q95], [q25,q75], [q25,q95], [q05,q75]
    intervals = [
        (0, 4, "τ[5%~95%]  (90% PI)"),
        (1, 3, "τ[25%~75%] (50% PI)"),
        (1, 4, "τ[25%~95%] (70% PI)"),
        (0, 3, "τ[5%~75%]  (70% PI)"),
    ]
    labels, picps, mpiws = [], [], []
    for lo_i, hi_i, lbl in intervals:
        lo, hi = yp[:, lo_i], yp[:, hi_i]
        picp_v = float(np.mean((yt >= lo) & (yt <= hi)) * 100)
        mpiw_v = float(np.mean(hi - lo))
        labels.append(lbl); picps.append(picp_v); mpiws.append(mpiw_v)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_p = ["#8E44AD","#2980B9","#27AE60","#E67E22"]
    axes[0].barh(labels, picps, color=colors_p, alpha=0.85)
    axes[0].axvline(90, color="red", lw=1.5, ls="--", label="90% target")
    axes[0].set_title("PICP — Prediction Interval Coverage", fontsize=11)
    axes[0].set_xlabel("Coverage (%)"); axes[0].legend(fontsize=9)
    for i, v in enumerate(picps):
        axes[0].text(v+0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    axes[1].barh(labels, mpiws, color=colors_p, alpha=0.85)
    axes[1].set_title("MPIW — Mean Prediction Interval Width", fontsize=11)
    axes[1].set_xlabel("Width")
    for i, v in enumerate(mpiws):
        axes[1].text(v+0.0005, i, f"{v:.5f}", va="center", fontsize=9)
    plt.suptitle("Coverage-Width Trade-off  (Val Set)", fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig(fig, "7th_004_coverage_width.png")


def plot_w7_all_quantiles_scatter(r7: dict) -> None:
    """그림 10: 각 τ의 예측값 vs 실제값 산점도 (2×3 서브플롯)."""
    yp = r7["yp_val"]
    yt = r7["yt_val"].flatten()
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for i, (tau, ax) in enumerate(zip(QUANTILES, axes)):
        yp_i = yp[:, i]
        lim  = max(abs(yt).max(), abs(yp_i).max()) * 1.1
        ax.scatter(yt, yp_i, s=6, alpha=0.35, color=COLORS[tau])
        ax.plot([-lim,lim],[-lim,lim],"k--",lw=1,alpha=0.5)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
        ax.set_title(f"τ={tau}", fontsize=10)
        ax.set_xlabel("Actual", fontsize=8); ax.set_ylabel("Predicted", fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Scatter: Actual vs Predicted for Each Quantile  (Val Set)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig(fig, "7th_005_quantile_scatter.png")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    splits = load_windows()

    # ── 6차시 ─────────────────────────────────────────────────
    r6 = run_week6(splits, device)
    print("\n  [6차시 시각화 생성]")
    plot_w6_training(r6)
    plot_w6_pred(r6)
    plot_w6_loss_compare(r6)
    plot_w6_baseline_compare(r6)
    plot_w6_residual(r6)

    # ── 7차시 ─────────────────────────────────────────────────
    r7 = run_week7(splits, device)
    print("\n  [7차시 시각화 생성]")
    plot_w7_training(r7)
    plot_w7_prediction_interval(r7)
    plot_w7_quantile_each(r7)
    plot_w7_coverage(r7)
    plot_w7_all_quantiles_scatter(r7)

    import pickle
    with open(MODEL_DIR / "week67_results.pkl", "wb") as f:
        pickle.dump({"r6": r6, "r7": r7}, f)
    print("\n  결과 저장: data/models/week67_results.pkl")

    print(f"\n{'='*55}")
    print("  6~7차시 완료 요약")
    print(f"{'='*55}")
    print(f"  [6차시] τ=0.5 Val MAE  = {r6['val_metrics']['mae']:.6f}")
    print(f"          τ=0.5 Val RMSE = {r6['val_metrics']['rmse']:.6f}")
    print(f"          DirAcc         = {r6['val_metrics']['dir_acc']:.2f}%")
    print(f"          Baseline 대비  = {r6['improvement']:+.2f}%")
    print(f"  [7차시] PICP(95%)      = {r7['picp_95']:.2f}%")
    print(f"          MPIW           = {r7['mpiw']:.6f}")
    print(f"{'='*55}\n")
    return r6, r7


if __name__ == "__main__":
    main()
