"""
notebooks/11_final_compare.py
────────────────────────────────────────────────────────────────────────────
13차시: 5개 모델 종합 비교 분석

■ 비교 모델
  1. Baseline LSTM (MSE, W5)
  2. Multi-Q LSTM (τ=5개, W7)
  3. Optuna LSTM (최적화, W8)
  4. TFT (W10)
  5. Best Overall (CRPS 기준 선정)

■ 비교 차원
  · Point Prediction: MAE, RMSE
  · Probabilistic:    CRPS, PICP, MPIW, Winkler
  · Risk:             VaR Failure, Kupiec, Christoffersen
  · Efficiency:       Params, Training Time proxy
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys, warnings, pickle
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from config.settings import DATA_SPLIT, FIG_DIR
from src.evaluation.metrics import mae, rmse, directional_accuracy
from src.evaluation.quantile_metrics import picp, mpiw, winkler_score, kupiec_test
from src.losses.crps import crps_quantile
from src.evaluation.backtest import quantile_hit_rate

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

plt.rcParams.update({
    "figure.dpi": 130, "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_all():
    d = np.load(DATA_SPLIT/"windows.npz")
    y_val  = d["y_val"].flatten()
    y_test = d["y_test"].flatten()
    with open(MODEL_DIR/"week67_results.pkl","rb") as f:  r5  = pickle.load(f)
    with open(MODEL_DIR/"week9_results.pkl","rb") as f:  r9  = pickle.load(f)
    tft = torch.load(MODEL_DIR/"tft_best.pt", map_location="cpu", weights_only=False)
    return {"y_val":y_val,"y_test":y_test,"r5":r5,"r9":r9,"tft":tft}


def compute_all_metrics(data) -> dict:
    y_val  = data["y_val"]
    r5     = data["r5"]
    r9     = data["r9"]
    tft    = data["tft"]

    results = {}

    # Baseline LSTM (W5) — point prediction only
    yp5 = r5["r6"]["yp_val"].flatten()
    yt5 = r5["r6"]["yt_val"].flatten()
    results["Baseline LSTM"] = {
        "mae":  mae(yt5, yp5), "rmse": rmse(yt5, yp5),
        "dir":  directional_accuracy(yt5, yp5),
        "crps": None, "picp": None, "mpiw": None,
        "wink": None, "kup_p": None, "fail": None,
        "params": 54209, "n_quantiles": 1,
        "model_type": "Point",
    }

    # Multi-Q LSTM (W7) / Optuna LSTM (W8)
    for tag, label in [("Week7 Multi-Q","Multi-Q LSTM"),("Optuna Best (W8)","Optuna LSTM")]:
        yp = r9[tag]["yp_val"]; yt = y_val
        kup = kupiec_test(yt, yp[:,0])
        results[label] = {
            "mae":  mae(yt, yp[:,2]), "rmse": rmse(yt, yp[:,2]),
            "dir":  directional_accuracy(yt, yp[:,2]),
            "crps": crps_quantile(yp, yt, QUANTILES),
            "picp": picp(yt, yp[:,0], yp[:,4]),
            "mpiw": mpiw(yp[:,0], yp[:,4]),
            "wink": winkler_score(yt, yp[:,0], yp[:,4]),
            "kup_p": kup["p_value"], "fail": yp[:,0],
            "params": 54469 if label=="Multi-Q LSTM" else 87429,
            "n_quantiles": 5, "model_type": "Quantile",
        }

    # TFT (W10)
    yp_t = tft["yp_val"]; yt_t = tft["yt_val"].flatten() if hasattr(tft["yt_val"],"flatten") else y_val
    kup_t = kupiec_test(yt_t, yp_t[:,0])
    results["TFT"] = {
        "mae":  mae(yt_t, yp_t[:,2]), "rmse": rmse(yt_t, yp_t[:,2]),
        "dir":  directional_accuracy(yt_t, yp_t[:,2]),
        "crps": tft["crps_val"],
        "picp": tft["picp_val"],
        "mpiw": tft["mpiw_val"],
        "wink": winkler_score(yt_t, yp_t[:,0], yp_t[:,4]),
        "kup_p": kup_t["p_value"], "fail": yp_t[:,0],
        "params": 173747, "n_quantiles": 5, "model_type": "Attention+Quantile",
    }

    return results


def save_fig(fig, name):
    p = FIG_DIR/name; fig.savefig(p,dpi=130,bbox_inches="tight"); plt.close(fig); print(f"  [Fig] {p.name}")


def plot_full_comparison_table(results):
    rows = []
    for name, r in results.items():
        crps_s = f"{r['crps']:.4f}" if r["crps"] else "N/A"
        picp_s = f"{r['picp']:.1f}%" if r["picp"] else "N/A"
        mpiw_s = f"{r['mpiw']:.4f}" if r["mpiw"] else "N/A"
        kup_s  = f"{r['kup_p']:.3f}" if r["kup_p"] else "N/A"
        rows.append([name, r["model_type"],
                     f"{r['mae']:.4f}", f"{r['rmse']:.4f}", f"{r['dir']:.1f}%",
                     crps_s, picp_s, mpiw_s, kup_s,
                     f"{r['params']:,}"])
    fig, ax = plt.subplots(figsize=(18,4))
    ax.axis("off")
    headers=["Model","Type","MAE","RMSE","DirAcc","CRPS","PICP(95%)","MPIW","Kupiec p","Params"]
    tbl=ax.table(cellText=rows,colLabels=headers,cellLoc="center",loc="center",colColours=["#1A252F"]*10)
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.15,2.4)
    fc_list=["#D6EAF8","#EAFAF1","#EAFAF1","#FEF9E7","#FDEDEC"]
    for (r2,c2),cell in tbl.get_celld().items():
        if r2==0: cell.set_text_props(color="white",fontweight="bold")
        elif r2<=len(rows): cell.set_facecolor(fc_list[r2-1])
    ax.set_title("13차시 5-Model Final Comparison Table", fontsize=12, pad=10)
    plt.tight_layout(); save_fig(fig,"13th_001_final_table.png")


def plot_radar_chart(results):
    """레이더 차트로 5개 모델 다차원 비교."""
    categories = ["MAE↓","RMSE↓","CRPS↓","PICP↑","DirAcc↑"]
    N = len(categories)

    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={"polar":True})
    colors  = ["#2980B9","#27AE60","#E67E22","#8E44AD","#E74C3C"]
    angles  = [n/float(N)*2*np.pi for n in range(N)] + [0]

    # 정규화 (0~1, 낮은게 좋은건 뒤집기)
    def normalize(vals, reverse=False):
        mn,mx = min(vals),max(vals)
        n = [(v-mn)/(mx-mn+1e-10) for v in vals]
        return [1-v for v in n] if reverse else n

    # 모델별 값 추출
    mae_vals  = [r["mae"]  for r in results.values()]
    rmse_vals = [r["rmse"] for r in results.values()]
    crps_vals = [r["crps"] if r["crps"] else max([rr["crps"] for rr in results.values() if rr["crps"]]) for r in results.values()]
    picp_vals = [r["picp"] if r["picp"] else 0 for r in results.values()]
    dir_vals  = [r["dir"]  for r in results.values()]

    mae_n  = normalize(mae_vals,  reverse=True)
    rmse_n = normalize(rmse_vals, reverse=True)
    crps_n = normalize(crps_vals, reverse=True)
    picp_n = normalize(picp_vals, reverse=False)
    dir_n  = normalize(dir_vals,  reverse=False)

    for i,(name,c) in enumerate(zip(results.keys(),colors)):
        vals = [mae_n[i],rmse_n[i],crps_n[i],picp_n[i],dir_n[i]] + [mae_n[i]]
        ax.plot(angles, vals, "o-", lw=2, color=c, label=name, markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=c)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=11)
    ax.set_title("Multi-Dimensional Model Comparison\n(Normalized, outer=better)", fontsize=12, pad=20)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.35,1.15))
    plt.tight_layout(); save_fig(fig,"13th_002_radar_chart.png")


def plot_metric_bars(results):
    """CRPS·PICP·MPIW 막대 비교."""
    q_models = {k:v for k,v in results.items() if v["crps"]}
    names   = list(q_models.keys())
    short   = ["Multi-Q\nLSTM","Optuna\nLSTM","TFT"]
    crps_v  = [q_models[n]["crps"] for n in names]
    picp_v  = [q_models[n]["picp"] for n in names]
    mpiw_v  = [q_models[n]["mpiw"] for n in names]
    colors  = ["#27AE60","#E67E22","#8E44AD"]
    x=np.arange(len(names))

    fig, axes = plt.subplots(1,3, figsize=(14,5))
    for ax,vals,title,goal,c_list in zip(axes,
        [crps_v,picp_v,mpiw_v],
        ["CRPS (↓ better)","PICP 95% (↑ better, ≥90%)","MPIW (↓ better)"],
        [0.08,90,None],[colors,colors,colors]):
        bars=ax.bar(x,vals,color=c_list,alpha=0.85,width=0.5)
        if goal: ax.axhline(goal,color="red",ls="--",lw=1.5,label=f"Target ({goal})")
        ax.set_xticks(x); ax.set_xticklabels(short,fontsize=9)
        ax.set_title(title,fontsize=11)
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height()*1.01,f"{v:.4f}",ha="center",fontsize=8.5,fontweight="bold")
        if goal: ax.legend(fontsize=9)
        ax.grid(alpha=0.3,axis="y")
    plt.suptitle("Probabilistic Metrics — Quantile Models", fontsize=12, y=1.02)
    plt.tight_layout(); save_fig(fig,"13th_003_metric_bars.png")


def plot_params_vs_crps(results):
    """파라미터 수 vs CRPS scatter."""
    fig, ax = plt.subplots(figsize=(8,5))
    colors={"Baseline LSTM":"#2C3E50","Multi-Q LSTM":"#27AE60","Optuna LSTM":"#E67E22","TFT":"#8E44AD"}
    for name,r in results.items():
        crps = r["crps"] if r["crps"] else 0.15
        ax.scatter(r["params"]/1e3, crps, s=150, color=colors[name], zorder=5, label=name)
        ax.annotate(name.replace(" LSTM","").replace(" (","\n("), (r["params"]/1e3, crps),
                    textcoords="offset points", xytext=(6,4), fontsize=8.5)
    ax.axhline(0.08, color="red", ls="--", lw=1.5, alpha=0.7, label="CRPS Target (0.08)")
    ax.set_xlabel("Parameters (K)"); ax.set_ylabel("CRPS (Val)")
    ax.set_title("Model Complexity vs Probabilistic Accuracy\n(CRPS Target = 0.08)", fontsize=12)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig,"13th_004_params_vs_crps.png")


def plot_mae_improvement(results):
    """MAE 개선 워터폴 차트."""
    names  = list(results.keys())
    maes   = [results[n]["mae"] for n in names]
    base   = maes[0]
    imps   = [(base - m)/base*100 for m in maes]

    fig, axes = plt.subplots(1,2, figsize=(13,5))
    colors_b = ["#A84B2F","#27AE60","#27AE60","#8E44AD","#8E44AD"]
    short_n  = ["Baseline\n(W5)","Multi-Q\n(W7)","Optuna\n(W8)","TFT\n(W10)","TFT\nTest"]

    bars = axes[0].bar(short_n[:4], maes[:4], color=colors_b[:4], alpha=0.85, width=0.5)
    axes[0].axhline(base, color="gray", ls=":", lw=1.2)
    axes[0].set_title("MAE Comparison  (τ=0.50 prediction, Val)", fontsize=11)
    axes[0].set_ylabel("MAE"); axes[0].grid(alpha=0.3, axis="y")
    for b,v in zip(bars,maes[:4]): axes[0].text(b.get_x()+b.get_width()/2,v*1.005,f"{v:.4f}",ha="center",fontsize=8.5,fontweight="bold")

    imp_colors = ["gray" if v<=0 else "#27AE60" for v in imps[:4]]
    bars2 = axes[1].bar(short_n[:4], imps[:4], color=imp_colors, alpha=0.85, width=0.5)
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].axhline(15, color="red", ls="--", lw=1.5, label="Target (+15%)")
    axes[1].set_title("MAE Improvement vs Baseline (%)", fontsize=11)
    axes[1].set_ylabel("Improvement (%)"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, axis="y")
    for b,v in zip(bars2,imps[:4]): axes[1].text(b.get_x()+b.get_width()/2,v+0.2,f"{v:+.2f}%",ha="center",fontsize=8.5,fontweight="bold")

    plt.suptitle("MAE Analysis — Model Evolution (W5 → W10)", fontsize=12, y=1.02)
    plt.tight_layout(); save_fig(fig,"13th_005_mae_improvement.png")


if __name__ == "__main__":
    print(f"\n{'='*55}\n  13차시: 5개 모델 종합 비교 분석\n{'='*55}\n")
    data    = load_all()
    results = compute_all_metrics(data)

    print("  [성능 요약]")
    for name,r in results.items():
        crps_s = f"{r['crps']:.5f}" if r["crps"] else "N/A    "
        print(f"  {name:<18} MAE={r['mae']:.5f} CRPS={crps_s} PICP={r['picp'] or 'N/A'}")

    with open(MODEL_DIR/"final_compare.pkl","wb") as f:
        pickle.dump(results,f)
    print("\n  저장: data/models/final_compare.pkl")

    print("\n  시각화 생성 중 ...")
    plot_full_comparison_table(results)
    plot_radar_chart(results)
    plot_metric_bars(results)
    plot_params_vs_crps(results)
    plot_mae_improvement(results)
    print("  13차시 완료!")
