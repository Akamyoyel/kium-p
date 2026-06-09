"""
notebooks/08_tft.py
────────────────────────────────────────────────────────────────────────────
10차시: Temporal Fusion Transformer (TFT) 학습 스크립트

■ 실행 방법
  python notebooks/08_tft.py

■ 저장 결과
  data/models/tft_best.pt
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

from config.settings import DATA_SPLIT, FIG_DIR
from src.models.tft import TemporalFusionTransformer
from src.losses.pinball import MultiQuantileLoss
from src.losses.crps import crps_quantile
from src.evaluation.quantile_metrics import picp, mpiw, kupiec_test

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(exist_ok=True)

QUANTILES  = [0.05, 0.25, 0.50, 0.75, 0.95]
FEAT_NAMES = ['log_ret','lag1','lag5','lag12','sq','vol20','vol60','ratio','lev',
              'range','upper','lower','vlog','vchg','time']


def load_windows() -> dict:
    d = np.load(DATA_SPLIT / "windows.npz")
    return {k: torch.tensor(d[k]) for k in d.files}


def train(epochs: int = 80, patience: int = 15,
          hidden_size: int = 64, n_heads: int = 4,
          dropout: float = 0.15, lr: float = 8e-4,
          batch_size: int = 128) -> dict:
    """TFT 학습 메인 함수."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  10차시: Temporal Fusion Transformer 학습")
    print(f"  Device: {device}")
    print(f"{'='*55}\n")

    splits = load_windows()
    print(f"  X_train: {tuple(splits['X_train'].shape)}")
    print(f"  X_val  : {tuple(splits['X_val'].shape)}")

    tr_ld = DataLoader(TensorDataset(splits["X_train"], splits["y_train"]),
                       batch_size=batch_size, shuffle=True)
    vl_ld = DataLoader(TensorDataset(splits["X_val"],   splits["y_val"]),
                       batch_size=batch_size, shuffle=False)
    te_ld = DataLoader(TensorDataset(splits["X_test"],  splits["y_test"]),
                       batch_size=batch_size, shuffle=False)

    model = TemporalFusionTransformer(
        input_size=15, hidden_size=hidden_size,
        n_heads=n_heads, n_quantiles=5,
        dropout=dropout, seq_len=60,
    ).to(device)
    print(f"\n  모델: {model}")

    criterion = MultiQuantileLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=7
    )

    th, vh = [], []
    best_vl = 1e9; no_imp = 0; best_st = None

    for ep in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────
        model.train(); tl = 0.0
        for X, y in tr_ld:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(X)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * X.size(0)
        tl /= len(tr_ld.dataset)

        # ── Val ───────────────────────────────────────────────
        model.eval(); vl = 0.0
        with torch.no_grad():
            for X, y in vl_ld:
                X, y = X.to(device), y.to(device)
                out, _ = model(X)
                vl += criterion(out, y).item() * X.size(0)
        vl /= len(vl_ld.dataset)
        scheduler.step(vl)
        th.append(tl); vh.append(vl)

        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep{ep:3d}  tr={tl:.5f}  vl={vl:.5f}  best={best_vl:.5f}")
        if no_imp >= patience:
            print(f"  ⏹ Early stop @ ep{ep}")
            break

    # ── 평가 ──────────────────────────────────────────────────
    model.load_state_dict(best_st); model.eval()

    def get_preds_weights(ld):
        ps, ys, ws = [], [], []
        with torch.no_grad():
            for X, y in ld:
                X = X.to(device)
                out, w = model(X)
                ps.append(out.cpu().numpy())
                ys.append(y.numpy())
                ws.append(w.cpu().numpy())
        return (np.concatenate(ps),
                np.concatenate(ys).flatten(),
                np.concatenate(ws))

    yp_v, yt_v, w_v = get_preds_weights(vl_ld)
    yp_t, yt_t, w_t = get_preds_weights(te_ld)

    crps_v  = crps_quantile(yp_v, yt_v, QUANTILES)
    picp_v  = picp(yt_v, yp_v[:, 0], yp_v[:, 4])
    mpiw_v  = mpiw(yp_v[:, 0], yp_v[:, 4])
    kup     = kupiec_test(yt_v, yp_v[:, 0], alpha=0.05)
    crps_t  = crps_quantile(yp_t, yt_t, QUANTILES)
    mean_imp = w_v.mean(axis=(0, 1))   # (15,)

    print(f"\n{'='*55}")
    print(f"  TFT 결과")
    print(f"{'='*55}")
    print(f"  Val CRPS  = {crps_v:.6f}")
    print(f"  Val PICP  = {picp_v:.2f}%")
    print(f"  Val MPIW  = {mpiw_v:.6f}")
    print(f"  Test CRPS = {crps_t:.6f}")
    print(f"  Kupiec p  = {kup['p_value']:.4f}  {'✅' if not kup['reject_H0'] else '❌'}")
    print(f"\n  VSN 피처 중요도 Top-5:")
    for i in np.argsort(mean_imp)[::-1][:5]:
        print(f"    {FEAT_NAMES[i]:<10}: {mean_imp[i]:.4f}")

    # ── 저장 ──────────────────────────────────────────────────
    torch.save({
        "model_state":     best_st,
        "val_loss":        best_vl,
        "train_hist":      np.array(th),
        "val_hist":        np.array(vh),
        "yp_val":          yp_v,
        "yt_val":          yt_v,
        "yp_test":         yp_t,
        "yt_test":         yt_t,
        "vsn_weights_val": w_v,
        "mean_importance": mean_imp,
        "feat_names":      FEAT_NAMES,
        "crps_val":        crps_v,
        "picp_val":        picp_v,
        "mpiw_val":        mpiw_v,
        "crps_test":       crps_t,
        "kupiec":          kup,
    }, MODEL_DIR / "tft_best.pt")
    print(f"\n  저장: data/models/tft_best.pt")
    print(f"{'='*55}\n")

    return {
        "yp_val": yp_v, "yt_val": yt_v,
        "yp_test": yp_t, "yt_test": yt_t,
        "vsn_weights": w_v, "mean_importance": mean_imp,
        "crps_val": crps_v, "picp_val": picp_v,
        "crps_test": crps_t, "kupiec": kup,
    }


if __name__ == "__main__":
    train(epochs=80, patience=15)
