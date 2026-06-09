"""
notebooks/04_baseline_lstm.py
────────────────────────────────────────────────────────────────────────────
5차시: LSTM Baseline Point Prediction

■ 수행 내용 (PDF 5차시 기준)
  1. LSTM Point Prediction 모델 구현 (MSE 손실)
  2. 하이퍼파라미터 탐색 (LR, hidden_size, num_layers)
  3. MAE, RMSE 기준 성능 측정
  4. 학습 곡선 시각화 (Train/Val Loss)
  5. 예측값 vs 실제값 비교 시각화

■ 결과 저장
  · 모델 가중치: data/models/lstm_point_best.pt
  · 학습 히스토리: data/models/train_history.npy
  · 성능 보고서: reports/model_spec.md 업데이트
────────────────────────────────────────────────────────────────────────────
"""

import sys
import warnings
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

from config.settings import DATA_SPLIT, FIG_DIR
from src.models.lstm_point import LSTMPointPredictor, print_model_summary
from src.losses.mse_loss import MSELoss
from src.evaluation.metrics import regression_report

# ── 경로 ─────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ═══════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════════

def load_windows() -> dict:
    """windows.npz 로드 → PyTorch Tensor 반환."""
    path = DATA_SPLIT / "windows.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"windows.npz 없음 — 4차시 run_feature_pipeline() 먼저 실행하세요.\n{path}"
        )
    data = np.load(path)
    return {k: torch.tensor(data[k]) for k in data.files}


# ═══════════════════════════════════════════════════════════════
# 2. 학습 루프
# ═══════════════════════════════════════════════════════════════

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """단일 에폭 학습, 평균 Loss 반환."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        # Gradient Clipping — 폭발적 기울기 방지
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Val/Test 평가 — (loss, y_pred_np, y_true_np)."""
    model.eval()
    n_samples = len(loader.dataset)
    if n_samples == 0:
        return np.nan, np.array([]), np.array([])

    total_loss = 0.0
    preds, targets = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        preds.append(pred.cpu().numpy())
        targets.append(y_batch.cpu().numpy())
    avg_loss = total_loss / n_samples
    y_pred   = np.concatenate(preds).flatten()
    y_true   = np.concatenate(targets).flatten()
    return avg_loss, y_pred, y_true


# ═══════════════════════════════════════════════════════════════
# 3. 하이퍼파라미터 그리드 탐색
# ═══════════════════════════════════════════════════════════════

GRID = [
    {"lr": 1e-3, "hidden": 32,  "layers": 1, "label": "LR=1e-3,H=32,L=1"},
    {"lr": 1e-3, "hidden": 64,  "layers": 2, "label": "LR=1e-3,H=64,L=2"},
    {"lr": 5e-4, "hidden": 64,  "layers": 2, "label": "LR=5e-4,H=64,L=2"},
    {"lr": 1e-3, "hidden": 128, "layers": 2, "label": "LR=1e-3,H=128,L=2"},
]


def run_training(
    params:      dict,
    splits:      dict,
    device:      torch.device,
    epochs:      int   = 100,
    batch_size:  int   = 64,
    patience:    int   = 15,
) -> dict:
    """
    단일 설정으로 학습 실행.

    Returns
    -------
    dict  {label, train_hist, val_hist, best_val_loss,
           y_pred_val, y_true_val, y_pred_test, y_true_test, model_state}
    """
    # DataLoader
    train_ds = TensorDataset(splits["X_train"], splits["y_train"])
    val_ds   = TensorDataset(splits["X_val"],   splits["y_val"])
    test_ds  = TensorDataset(splits["X_test"],  splits["y_test"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    input_size = splits["X_train"].shape[2]
    model = LSTMPointPredictor(
        input_size  = input_size,
        hidden_size = params["hidden"],
        num_layers  = params["layers"],
        dropout     = 0.2,
    ).to(device)

    criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    train_hist, val_hist = [], []
    best_val_loss = float("inf")
    no_improve    = 0
    best_state    = None

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, y_pred_val, y_true_val = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        train_hist.append(tr_loss)
        val_hist.append(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{params['label']}]  Epoch {epoch:3d} | "
                  f"Train={tr_loss:.6f}  Val={vl_loss:.6f}  "
                  f"Best={best_val_loss:.6f}  LR={optimizer.param_groups[0]['lr']:.2e}")

        if no_improve >= patience:
            print(f"  ⏹ Early stop @ epoch {epoch}")
            break

    # Test 평가
    model.load_state_dict(best_state)
    _, y_pred_test, y_true_test = evaluate(model, test_loader, criterion, device)
    _, y_pred_val_best, _       = evaluate(model, val_loader,  criterion, device)

    return {
        "label":        params["label"],
        "train_hist":   np.array(train_hist),
        "val_hist":     np.array(val_hist),
        "best_val_loss":best_val_loss,
        "y_pred_val":   y_pred_val_best,
        "y_true_val":   y_true_val,
        "y_pred_test":  y_pred_test,
        "y_true_test":  y_true_test,
        "model_state":  best_state,
        "params":       params,
    }


# ═══════════════════════════════════════════════════════════════
# 4. 시각화
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(results: list[dict]) -> None:
    """그림 1: 모든 설정의 Train/Val Loss 곡선 비교."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#20808D", "#A84B2F", "#1B474D", "#FFC553"]

    for r, c in zip(results, colors):
        axes[0].plot(r["train_hist"], lw=1.2, label=r["label"], color=c)
        axes[1].plot(r["val_hist"],   lw=1.2, label=r["label"], color=c)

    axes[0].set_title("Training Loss (MSE)  —  All Configs", fontsize=12)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].legend(fontsize=7.5); axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Loss (MSE)  —  All Configs", fontsize=12)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE Loss")
    axes[1].legend(fontsize=7.5); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = FIG_DIR / "5th_001_training_curves.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [Fig 1] 저장: {p.name}")


def plot_pred_vs_true(best: dict) -> None:
    """그림 2: 최적 모델의 예측값 vs 실제값 (Val set)."""
    y_pred = best["y_pred_val"]
    y_true = best["y_true_val"]
    n      = len(y_true)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 시계열 비교
    axes[0].plot(range(n), y_true, lw=1.0, color="#20808D", label="Actual",    alpha=0.9)
    axes[0].plot(range(n), y_pred, lw=1.0, color="#A84B2F", label="Predicted", alpha=0.8, ls="--")
    axes[0].set_title(f"Actual vs Predicted  ({best['label']})\nVal Set  (n={n})", fontsize=11)
    axes[0].set_xlabel("Time Step"); axes[0].set_ylabel("Log-Return")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # 산점도
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.1
    axes[1].scatter(y_true, y_pred, s=10, alpha=0.5, color="#20808D")
    axes[1].plot([-lim, lim], [-lim, lim], "r--", lw=1.5, label="Perfect Fit")
    axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
    axes[1].set_title("Scatter: Actual vs Predicted  (Val Set)", fontsize=11)
    axes[1].set_xlabel("Actual Log-Return")
    axes[1].set_ylabel("Predicted Log-Return")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = FIG_DIR / "5th_002_pred_vs_true.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [Fig 2] 저장: {p.name}")


def plot_performance_table(results: list[dict]) -> None:
    """그림 3: 설정별 Val 성능 지표 비교 막대그래프."""
    from src.evaluation.metrics import mae, rmse, directional_accuracy

    labels, maes, rmses, daccs = [], [], [], []
    for r in results:
        labels.append(r["label"].replace(",", "\n"))
        maes.append(mae(r["y_true_val"], r["y_pred_val"]))
        rmses.append(rmse(r["y_true_val"], r["y_pred_val"]))
        daccs.append(directional_accuracy(r["y_true_val"], r["y_pred_val"]))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, vals, title, color in zip(
        axes,
        [maes, rmses, daccs],
        ["MAE (↓ Better)", "RMSE (↓ Better)", "Directional Acc. (↑ Better)"],
        ["#20808D", "#A84B2F", "#27AE60"],
    ):
        bars = ax.bar(x, vals, color=color, alpha=0.8, width=0.55)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(title, fontsize=11)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{v:.5f}" if title != "Directional Acc. (↑ Better)" else f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.suptitle("Hyperparameter Search  —  Val Set Performance (Week 5)", fontsize=12, y=1.01)
    plt.tight_layout()
    p = FIG_DIR / "5th_003_hp_comparison.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [Fig 3] 저장: {p.name}")


def plot_residual_analysis(best: dict) -> None:
    """그림 4: 잔차 분석 (잔차 분포 + 잔차 시계열)."""
    y_pred = best["y_pred_val"]
    y_true = best["y_true_val"]
    resid  = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 잔차 히스토그램
    axes[0].hist(resid, bins=40, density=True, alpha=0.7, color="#20808D")
    mu, sig = resid.mean(), resid.std()
    x_range = np.linspace(resid.min(), resid.max(), 300)
    from scipy.stats import norm
    axes[0].plot(x_range, norm.pdf(x_range, mu, sig), "r--", lw=2, label=f"Normal(μ={mu:.5f})")
    axes[0].set_title(f"Residual Distribution  ({best['label']})", fontsize=11)
    axes[0].set_xlabel("Residual"); axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # 잔차 시계열
    axes[1].plot(resid, lw=0.8, color="#A84B2F", alpha=0.8)
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].axhline( 2*sig, color="#FFC553", lw=1, ls="--", label=f"±2σ = ±{2*sig:.5f}")
    axes[1].axhline(-2*sig, color="#FFC553", lw=1, ls="--")
    axes[1].set_title("Residual Time Series", fontsize=11)
    axes[1].set_xlabel("Time Step"); axes[1].set_ylabel("Residual")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = FIG_DIR / "5th_004_residual.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [Fig 4] 저장: {p.name}")


# ═══════════════════════════════════════════════════════════════
# 5. 메인 실행
# ═══════════════════════════════════════════════════════════════

def run_baseline(epochs: int = 100, batch_size: int = 64) -> dict:
    """
    5차시 전체 파이프라인 실행.

    Usage
    -----
    from notebooks.baseline_lstm import run_baseline
    best = run_baseline(epochs=100)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  5차시: LSTM Baseline Point Prediction")
    print(f"  Device: {device}")
    print(f"{'='*55}")

    # 1. 데이터 로드
    splits = load_windows()
    X_shape = splits["X_train"].shape
    print(f"\n  Dataset:  X_train={tuple(X_shape)}  y_train={tuple(splits['y_train'].shape)}")
    print(f"            X_val  ={tuple(splits['X_val'].shape)}")
    print(f"            X_test ={tuple(splits['X_test'].shape)}")

    # 2. 모델 요약 (대표 설정)
    sample_model = LSTMPointPredictor(input_size=X_shape[2])
    print_model_summary(sample_model)

    # 3. 하이퍼파라미터 탐색
    results = []
    for params in GRID:
        print(f"\n── 설정: {params['label']} ──")
        res = run_training(params, splits, device, epochs=epochs, batch_size=batch_size)
        results.append(res)

    # 4. 최적 모델 선택 (Val Loss 최소)
    best = min(results, key=lambda r: r["best_val_loss"])
    print(f"\n{'─'*55}")
    print(f"  ✅ 최적 설정: {best['label']}")
    print(f"     Best Val MSE : {best['best_val_loss']:.6f}")
    print(f"{'─'*55}")

    # 5. 최종 성능 측정 (Val + Test)
    val_metrics  = regression_report(best["y_true_val"],  best["y_pred_val"],  "Best Model  — Val Set")
    test_metrics = regression_report(best["y_true_test"], best["y_pred_test"], "Best Model  — Test Set")

    # 6. 모델 저장
    torch.save({
        "model_state": best["model_state"],
        "params":      best["params"],
        "val_metrics": val_metrics,
        "test_metrics":test_metrics,
    }, MODEL_DIR / "lstm_point_best.pt")
    np.save(MODEL_DIR / "train_history.npy",
            {"train": best["train_hist"], "val": best["val_hist"]})
    print(f"\n  모델 저장: data/models/lstm_point_best.pt")

    # 7. 시각화
    print("\n  시각화 생성:")
    plot_training_curves(results)
    plot_pred_vs_true(best)
    plot_performance_table(results)
    plot_residual_analysis(best)

    print(f"\n{'='*55}")
    print("  5차시 완료!")
    print(f"  → 6차시 준비: Quantile LSTM (τ=0.5) 구현 예정")
    print(f"     Baseline Val MAE  = {val_metrics['mae']:.6f}")
    print(f"     Baseline Val RMSE = {val_metrics['rmse']:.6f}")
    print(f"{'='*55}\n")

    return {"best": best, "all_results": results,
            "val_metrics": val_metrics, "test_metrics": test_metrics}


if __name__ == "__main__":
    run_baseline(epochs=100)
