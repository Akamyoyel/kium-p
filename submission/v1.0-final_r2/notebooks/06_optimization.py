"""
notebooks/06_optimization.py
────────────────────────────────────────────────────────────────────────────
8차시: Optuna 하이퍼파라미터 자동 최적화

■ 탐색 파라미터 (Search Space)
  · lr          : [1e-4, 5e-3] log scale
  · hidden_size : [32, 64, 128, 256]
  · num_layers  : [1, 2, 3]
  · dropout     : [0.1, 0.4]
  · batch_size  : [64, 128, 256]

■ 최적화 목표
  · Val MultiQuantileLoss 최소화
  · 20 trial (시간 제약 — GPU 환경에서는 100+)

■ 추가 기법
  · Dropout 강화: Regularization (과적합 방지)
  · LayerNorm: 학습 안정화
  · ReduceLROnPlateau: 학습률 동적 감소
  · Early Stopping: patience=15

■ 결과 저장
  · data/models/optuna_best.pt
  · data/models/optuna_study.pkl
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys, warnings, pickle
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
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history as _plot_hist,
    plot_param_importances    as _plot_imp,
)

from config.settings import DATA_SPLIT, FIG_DIR
from src.models.lstm_quantile import LSTMQuantilePredictor
from src.losses.pinball import MultiQuantileLoss

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "font.family": "DejaVu Sans",
    "axes.unicode_minus": False, "axes.spines.top": False,
    "axes.spines.right": False,
})

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
PICP_TARGET = 90.0


def crossing_penalty(pred: torch.Tensor) -> torch.Tensor:
    """Quantile crossing 방지 페널티 (단조 증가 제약 유도)."""
    diffs = pred[:, 1:] - pred[:, :-1]
    return torch.relu(-diffs).mean()


# ═══════════════════════════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════════════════════════

def load_windows() -> dict:
    d = np.load(DATA_SPLIT / "windows.npz")
    return {k: torch.tensor(d[k]) for k in d.files}


# ═══════════════════════════════════════════════════════════════
# 단일 Trial 학습
# ═══════════════════════════════════════════════════════════════

def train_and_eval(params: dict, splits: dict, device: torch.device,
                   epochs: int = 60, patience: int = 12) -> float:
    """지정 파라미터로 학습 후 복합 점수(낮을수록 좋음) 반환."""
    tr_ld = DataLoader(TensorDataset(splits["X_train"], splits["y_train"]),
                       batch_size=params["batch_size"], shuffle=True)
    vl_ld = DataLoader(TensorDataset(splits["X_val"],   splits["y_val"]),
                       batch_size=params["batch_size"], shuffle=False)

    model = LSTMQuantilePredictor(
        input_size  = 15,
        hidden_size = params["hidden_size"],
        num_layers  = params["num_layers"],
        dropout     = params["dropout"],
        n_quantiles = 5,
    ).to(device)

    criterion = MultiQuantileLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=7
    )

    best_score = 1e9
    no_imp = 0

    lambda_cross = 0.5
    lambda_cov = 0.5

    for ep in range(1, epochs + 1):
        model.train()
        for X, y in tr_ld:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            pinball = criterion(pred, y)
            loss = pinball + lambda_cross * crossing_penalty(pred)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval(); vl = 0.0; ps, ys = [], []
        with torch.no_grad():
            for X, y in vl_ld:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                vl += criterion(pred, y).item() * X.size(0)
                ps.append(pred.cpu().numpy())
                ys.append(y.cpu().numpy())
        vl /= len(vl_ld.dataset)
        scheduler.step(vl)

        yp = np.concatenate(ps)
        yt = np.concatenate(ys).flatten()
        picp_95 = float(np.mean((yt >= yp[:, 0]) & (yt <= yp[:, 4])) * 100)
        crps_approx = 2.0 * vl
        coverage_penalty = abs(picp_95 - PICP_TARGET) / 100.0
        score = crps_approx + lambda_cov * coverage_penalty

        if score < best_score:
            best_score = score
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    return best_score


# ═══════════════════════════════════════════════════════════════
# Optuna Objective
# ═══════════════════════════════════════════════════════════════

def make_objective(splits: dict, device: torch.device):
    """Optuna objective 함수 생성."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr":          trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
            "num_layers":  trial.suggest_int("num_layers", 1, 3),
            "dropout":     trial.suggest_float("dropout", 0.1, 0.4),
            "batch_size":  trial.suggest_categorical("batch_size", [64, 128, 256]),
        }
        return train_and_eval(params, splits, device, epochs=60, patience=12)
    return objective


# ═══════════════════════════════════════════════════════════════
# Optuna 탐색 실행
# ═══════════════════════════════════════════════════════════════

def run_optuna(n_trials: int = 20) -> dict:
    """
    8차시 Optuna 탐색 전체 실행.

    Returns
    -------
    dict — {study, best_params, best_value, best_model_state}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  8차시: Optuna 하이퍼파라미터 최적화")
    print(f"  Device: {device}  |  Trials: {n_trials}")
    print(f"{'='*55}\n")

    splits = load_windows()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    print(f"  탐색 시작 (n_trials={n_trials}) ...\n")
    study.optimize(
        make_objective(splits, device),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_p = study.best_params
    best_v = study.best_value
    print(f"  탐색 완료!")
    print(f"  최적 Val Loss : {best_v:.6f}")
    print(f"  최적 파라미터:")
    for k, v in best_p.items():
        print(f"    {k:15s}: {v}")

    # 최적 파라미터로 최종 학습 (더 많은 에폭)
    print(f"\n  최적 파라미터로 최종 학습 (epochs=100) ...")
    best_state, best_yp_val, best_yt_val, best_yp_test, best_yt_test = \
        _final_train(best_p, splits, device, epochs=100)

    # 저장
    with open(MODEL_DIR / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)
    torch.save({
        "model_state": best_state,
        "best_params": best_p,
        "best_val_loss": best_v,
        "yp_val": best_yp_val, "yt_val": best_yt_val,
        "yp_test": best_yp_test, "yt_test": best_yt_test,
    }, MODEL_DIR / "optuna_best.pt")
    print("  저장: data/models/optuna_study.pkl")
    print("  저장: data/models/optuna_best.pt")

    return {
        "study": study, "best_params": best_p,
        "best_value": best_v, "best_state": best_state,
        "yp_val": best_yp_val, "yt_val": best_yt_val,
        "yp_test": best_yp_test, "yt_test": best_yt_test,
    }


def _final_train(params: dict, splits: dict, device: torch.device,
                 epochs: int = 100):
    """최적 파라미터로 최종 학습 — 모델 상태 및 예측값 반환."""
    tr_ld = DataLoader(TensorDataset(splits["X_train"], splits["y_train"]),
                       batch_size=params["batch_size"], shuffle=True)
    vl_ld = DataLoader(TensorDataset(splits["X_val"],   splits["y_val"]),
                       batch_size=params["batch_size"], shuffle=False)
    te_ld = DataLoader(TensorDataset(splits["X_test"],  splits["y_test"]),
                       batch_size=params["batch_size"], shuffle=False)

    model = LSTMQuantilePredictor(
        input_size=15, hidden_size=params["hidden_size"],
        num_layers=params["num_layers"], dropout=params["dropout"], n_quantiles=5,
    ).to(device)
    criterion = MultiQuantileLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=7)

    lambda_cross = 0.5

    best_vl = 1e9; no_imp = 0; best_st = None; th, vh = [], []
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for X, y in tr_ld:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y) + lambda_cross * crossing_penalty(pred)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * X.size(0)
        tl /= len(tr_ld.dataset)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for X, y in vl_ld:
                X, y = X.to(device), y.to(device)
                vl += criterion(model(X), y).item() * X.size(0)
        vl /= len(vl_ld.dataset)
        th.append(tl); vh.append(vl)
        scheduler.step(vl)

        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep % 20 == 0:
            print(f"    Ep{ep:3d}  tr={tl:.5f}  vl={vl:.5f}  best={best_vl:.5f}")
        if no_imp >= 15:
            print(f"    ⏹ Early stop @ ep{ep}")
            break

    model.load_state_dict(best_st)
    model.eval()

    def get_preds(ld):
        if len(ld.dataset) == 0:
            return np.empty((0, 5)), np.empty((0,))
        ps, ys = [], []
        with torch.no_grad():
            for X, y in ld:
                ps.append(model(X.to(device)).cpu().numpy())
                ys.append(y.numpy())
        return np.concatenate(ps), np.concatenate(ys).flatten()

    yp_v, yt_v = get_preds(vl_ld)
    yp_t, yt_t = get_preds(te_ld)
    return best_st, yp_v, yt_v, yp_t, yt_t


# ═══════════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════════

def save_fig(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [Fig] 저장: {p.name}")


def plot_optimization_history(study) -> None:
    """그림 1: Optuna 탐색 히스토리."""
    trials = study.trials
    vals   = [t.value for t in trials if t.value is not None]
    best_so_far = np.minimum.accumulate(vals)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(range(len(vals)), vals, s=18, alpha=0.6, color="#2980B9",
                    label="Trial Value", zorder=3)
    axes[0].plot(range(len(best_so_far)), best_so_far, lw=2, color="#E74C3C",
                 label="Best so far")
    axes[0].set_title("Optuna Optimization History", fontsize=12)
    axes[0].set_xlabel("Trial #"); axes[0].set_ylabel("Val MultiQuantileLoss")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[0].axhline(study.best_value, color="#27AE60", ls="--", lw=1.5, alpha=0.7)

    # 파라미터별 Best Trial 값
    bp = study.best_params
    keys = list(bp.keys())
    axes[1].axis("off")
    table_data = [[k, str(v)] for k, v in bp.items()]
    tbl = axes[1].table(cellText=table_data,
                         colLabels=["Parameter", "Best Value"],
                         cellLoc="center", loc="center",
                         colColours=["#2C3E50", "#2C3E50"])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1.5, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
    axes[1].set_title(f"Best Parameters\n(Val Loss = {study.best_value:.6f})", fontsize=11)

    plt.tight_layout()
    save_fig(fig, "8th_001_optuna_history.png")


def plot_param_importance(study) -> None:
    """그림 2: 파라미터 중요도."""
    try:
        importances = optuna.importance.get_param_importances(study)
        params_list = list(importances.keys())
        vals_list   = list(importances.values())
    except Exception:
        # trial 수 부족시 균등 분배
        bp = study.best_params
        params_list = list(bp.keys())
        vals_list   = [1.0/len(params_list)] * len(params_list)

    colors = ["#20808D","#A84B2F","#8E44AD","#27AE60","#E67E22"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(params_list, vals_list,
                   color=colors[:len(params_list)], alpha=0.85)
    for bar, v in zip(bars, vals_list):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_title("Hyperparameter Importance  (Optuna FanovaImportance)", fontsize=11)
    ax.set_xlabel("Importance Score"); ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    save_fig(fig, "8th_002_param_importance.png")


def plot_before_after(res_optuna: dict) -> None:
    """그림 3: 최적화 전후 Val 예측 구간 비교."""
    with open(MODEL_DIR / "week67_results.pkl", "rb") as f:
        r67 = pickle.load(f)
    yp_before = r67["r7"]["yp_val"]   # (754, 5) — 7차시 결과
    yt         = r67["r7"]["yt_val"].flatten()

    yp_after  = res_optuna["yp_val"]  # (754, 5) — Optuna 결과
    n = min(150, len(yt))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(n)
    for i, (yp, title) in enumerate([(yp_before, "Before Optimization (Week 7)"),
                                       (yp_after,  "After Optuna (Week 8)")]):
        axes[i].fill_between(x, yp[:n,0], yp[:n,4], alpha=0.2, color="#8E44AD",
                             label="95% PI")
        axes[i].fill_between(x, yp[:n,1], yp[:n,3], alpha=0.3, color="#2980B9",
                             label="50% PI")
        axes[i].plot(x, yp[:n,2], lw=1.2, color="#2980B9", label="Median")
        axes[i].plot(x, yt[:n],   lw=0.9, color="#A84B2F", alpha=0.9, label="Actual")
        axes[i].set_title(title, fontsize=11)
        axes[i].set_ylabel("Scaled Log-Return")
        axes[i].legend(fontsize=8, loc="upper right"); axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    plt.suptitle("Prediction Interval: Before vs After Optimization", fontsize=12, y=1.01)
    plt.tight_layout()
    save_fig(fig, "8th_003_before_after_pi.png")


def plot_trial_scatter(study) -> None:
    """그림 4: 파라미터 값 vs Trial Loss 산점도."""
    trials = [t for t in study.trials if t.value is not None]
    params_to_plot = ["lr", "hidden_size", "dropout"]
    fig, axes = plt.subplots(1, len(params_to_plot), figsize=(14, 5))
    colors_t = plt.cm.RdYlGn_r(
        np.linspace(0, 1, len(trials))
    )
    vals_arr = np.array([t.value for t in trials])
    for ax, pname in zip(axes, params_to_plot):
        xs = [t.params.get(pname, None) for t in trials]
        ys = [t.value for t in trials]
        sc = ax.scatter(xs, ys, c=vals_arr, cmap="RdYlGn_r", s=30, alpha=0.8)
        ax.set_title(f"{pname} vs Val Loss", fontsize=10)
        ax.set_xlabel(pname); ax.set_ylabel("Val Loss")
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Val Loss")
    plt.suptitle("Parameter vs Val Loss Scatter  (Optuna Trials)", fontsize=11, y=1.02)
    plt.tight_layout()
    save_fig(fig, "8th_004_trial_scatter.png")


def plot_optuna_pred_interval(res_optuna: dict) -> None:
    """그림 5: Optuna 최적 모델의 예측 구간."""
    yp = res_optuna["yp_val"]
    yt = res_optuna["yt_val"].flatten()
    n  = min(200, len(yt))
    x  = np.arange(n)

    # PICP 계산
    picp_95 = float(np.mean((yt[:n] >= yp[:n,0]) & (yt[:n] <= yp[:n,4])) * 100)
    mpiw_95 = float(np.mean(yp[:n,4] - yp[:n,0]))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(x, yp[:n,0], yp[:n,4], alpha=0.18, color="#8E44AD",
                    label="95% PI (τ=0.05~0.95)")
    ax.fill_between(x, yp[:n,1], yp[:n,3], alpha=0.30, color="#2980B9",
                    label="50% PI (τ=0.25~0.75)")
    ax.plot(x, yp[:n,2], lw=1.2, color="#2980B9", label="Median (τ=0.50)")
    ax.plot(x, yt[:n],   lw=0.9, color="#A84B2C", alpha=0.9, label="Actual")
    ax.set_title("Optuna Best Model — Prediction Interval (Val Set, first 200 steps)",
                 fontsize=12)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Scaled Log-Return")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
    ax.text(0.02, 0.97,
        f"PICP(95%) = {picp_95:.1f}%\nMPIW = {mpiw_95:.5f}",
        transform=ax.transAxes, fontsize=9, va="top", fontfamily="monospace",
        bbox=dict(fc="white", ec="#ccc", alpha=0.9, boxstyle="round,pad=0.3"))
    plt.tight_layout()
    save_fig(fig, "8th_005_optuna_pred_interval.png")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

def main(n_trials: int = 20):
    res = run_optuna(n_trials=n_trials)
    study = res["study"]

    print("\n  [8차시 시각화 생성]")
    plot_optimization_history(study)
    plot_param_importance(study)
    plot_before_after(res)
    plot_trial_scatter(study)
    plot_optuna_pred_interval(res)

    print(f"\n{'='*55}")
    print("  8차시 완료!")
    print(f"  최적 Val Loss : {res['best_value']:.6f}")
    print(f"  → 9차시: CRPS·VaR Coverage·Kupiec Test 평가 예정")
    print(f"{'='*55}\n")
    return res


if __name__ == "__main__":
    main(n_trials=20)
