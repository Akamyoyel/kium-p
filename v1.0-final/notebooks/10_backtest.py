"""
notebooks/10_backtest.py
────────────────────────────────────────────────────────────────────────────
12차시: VaR 백테스트 종합 분석

■ 수행 내용
  1. 3개 모델(Multi-Q, Optuna, TFT)의 VaR(95%) 실패율 비교
  2. Kupiec POF Test — 통계적 VaR 적절성 검정
  3. Christoffersen Interval Forecast Test — 독립성 검정
  4. 포트폴리오 P&L 시뮬레이션 비교
  5. 분위수별 Hit Rate 비교
  6. 결과 저장: data/models/backtest_results.pkl
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
from scipy import stats

from config.settings import DATA_SPLIT, FIG_DIR
from src.evaluation.quantile_metrics import picp, mpiw, kupiec_test, var_coverage
from src.evaluation.backtest import quantile_hit_rate, var_backtest_summary, portfolio_pnl
from src.losses.crps import crps_quantile

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

plt.rcParams.update({
    "figure.dpi": 130, "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def christoffersen_test(y_true: np.ndarray, var_lower: np.ndarray,
                        alpha: float = 0.05) -> dict:
    """
    Christoffersen (1998) Interval Forecast Test.
    H₀: 실패가 독립적으로 발생 (클러스터링 없음)
    LR_ind ~ χ²(1)
    """
    y    = np.asarray(y_true).flatten()
    var  = np.asarray(var_lower).flatten()
    hits = (y < var).astype(int)
    n    = len(hits)
    T00 = T01 = T10 = T11 = 0
    for i in range(1, n):
        if hits[i-1]==0 and hits[i]==0: T00+=1
        elif hits[i-1]==0 and hits[i]==1: T01+=1
        elif hits[i-1]==1 and hits[i]==0: T10+=1
        else: T11+=1
    pi01 = T01/(T00+T01+1e-10)
    pi11 = T11/(T10+T11+1e-10)
    pi   = (T01+T11)/(T00+T01+T10+T11+1e-10)
    def safe_log(x): return np.log(max(x, 1e-10))
    lr = -2*(T00*safe_log(1-pi) + T01*safe_log(pi) + T10*safe_log(1-pi) + T11*safe_log(pi)) + \
          2*(T00*safe_log(1-pi01) + T01*safe_log(pi01+1e-10) + T10*safe_log(1-pi11) + T11*safe_log(pi11+1e-10))
    p_val = float(1 - stats.chi2.cdf(max(lr, 0), df=1))
    return {"T00":T00,"T01":T01,"T10":T10,"T11":T11,"pi01":pi01,"pi11":pi11,
            "LR_ind":float(lr),"p_value":p_val,"reject_H0": p_val<0.05}


def load_model_results() -> dict:
    d = np.load(DATA_SPLIT/"windows.npz")
    y_val  = d["y_val"].flatten()
    y_test = d["y_test"].flatten()
    with open(MODEL_DIR/"week9_results.pkl","rb") as f: r9 = pickle.load(f)
    tft = torch.load(MODEL_DIR/"tft_best.pt", map_location="cpu", weights_only=False)
    return {
        "y_val": y_val, "y_test": y_test,
        "Multi-Q LSTM (W7)": {"yp_val": r9["Week7 Multi-Q"]["yp_val"], "yp_test": r9["Week7 Multi-Q"]["yp_test"]},
        "Optuna LSTM (W8)":  {"yp_val": r9["Optuna Best (W8)"]["yp_val"], "yp_test": r9["Optuna Best (W8)"]["yp_test"]},
        "TFT (W10)":         {"yp_val": tft["yp_val"]},
    }


def run_backtest() -> dict:
    print(f"\n{'='*55}\n  12차시: VaR 백테스트 종합 분석\n{'='*55}\n")
    data    = load_model_results()
    y_val   = data["y_val"]
    results = {}
    models  = ["Multi-Q LSTM (W7)", "Optuna LSTM (W8)", "TFT (W10)"]

    for name in models:
        yp  = data[name]["yp_val"]
        var_lo = yp[:, 0]
        kupiec = kupiec_test(y_val, var_lo)
        christ = christoffersen_test(y_val, var_lo)
        bt     = var_backtest_summary(y_val, var_lo)
        pnl_r  = portfolio_pnl(y_val, var_lo)
        hit    = quantile_hit_rate(y_val, yp, QUANTILES)
        crps_v = crps_quantile(yp, y_val, QUANTILES)
        picp_v = picp(y_val, var_lo, yp[:,4])
        mpiw_v = mpiw(var_lo, yp[:,4])
        print(f"  [{name}]")
        print(f"    Failure Rate : {bt['failure_rate']:.2f}%  (Target 5%)")
        print(f"    Kupiec p     : {kupiec['p_value']:.4f}  {'OK' if not kupiec['reject_H0'] else 'FAIL'}")
        print(f"    Christoff. p : {christ['p_value']:.4f}  {'OK' if not christ['reject_H0'] else 'FAIL'}")
        print(f"    CRPS         : {crps_v:.5f}")
        print(f"    PICP(95%)    : {picp_v:.2f}%")
        results[name] = {"kupiec":kupiec,"christ":christ,"bt":bt,
                          "pnl":pnl_r,"hit":hit,"crps":crps_v,"picp":picp_v,
                          "mpiw":mpiw_v,"yp":yp}

    with open(MODEL_DIR/"backtest_results.pkl","wb") as f:
        pickle.dump({"results":results,"y_val":y_val},f)
    print("\n  저장: data/models/backtest_results.pkl")
    return results, y_val


def save_fig(fig, name):
    p = FIG_DIR/name; fig.savefig(p,dpi=130,bbox_inches="tight"); plt.close(fig)
    print(f"  [Fig] {p.name}")


def plot_failure_rates(results, y_val):
    models = list(results.keys())
    frates = [results[n]["bt"]["failure_rate"] for n in models]
    kupiec_p = [results[n]["kupiec"]["p_value"] for n in models]
    christ_p = [results[n]["christ"]["p_value"] for n in models]

    fig, axes = plt.subplots(1,3, figsize=(14,5))
    colors = ["#2980B9","#27AE60","#8E44AD"]
    x = np.arange(len(models)); short_names=["Multi-Q\n(W7)","Optuna\n(W8)","TFT\n(W10)"]

    bars = axes[0].bar(x, frates, color=colors, alpha=0.85, width=0.5)
    axes[0].axhline(5.0, color="red", ls="--", lw=1.8, label="Target (5%)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(short_names, fontsize=9)
    axes[0].set_title("VaR Failure Rate (%)", fontsize=11)
    for b,v in zip(bars,frates): axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f"{v:.2f}%", ha="center", fontsize=9, fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3, axis="y")

    bars2 = axes[1].bar(x, kupiec_p, color=colors, alpha=0.85, width=0.5)
    axes[1].axhline(0.05, color="red", ls="--", lw=1.8, label="α=0.05")
    axes[1].set_xticks(x); axes[1].set_xticklabels(short_names, fontsize=9)
    axes[1].set_title("Kupiec p-value\n(H₀: Failure Rate = 5%)", fontsize=11)
    for b,v in zip(bars2,kupiec_p): axes[1].text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, axis="y")

    bars3 = axes[2].bar(x, christ_p, color=colors, alpha=0.85, width=0.5)
    axes[2].axhline(0.05, color="red", ls="--", lw=1.8, label="α=0.05")
    axes[2].set_xticks(x); axes[2].set_xticklabels(short_names, fontsize=9)
    axes[2].set_title("Christoffersen p-value\n(H₀: Failures are Independent)", fontsize=11)
    for b,v in zip(bars3,christ_p): axes[2].text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3, axis="y")

    plt.suptitle("VaR Backtest Summary — 3 Models", fontsize=12, y=1.02)
    plt.tight_layout(); save_fig(fig, "12th_001_var_backtest.png")


def plot_pnl_comparison(results, y_val):
    models = list(results.keys())
    colors = ["#2980B9","#27AE60","#8E44AD"]
    fig, ax = plt.subplots(figsize=(14,5))
    x = np.arange(len(y_val))
    for name, c in zip(models, colors):
        pnl = results[name]["pnl"]["pnl"]
        ax.plot(x, pnl/1e6, lw=1.2, color=c, label=name, alpha=0.9)
    ax.set_title("Portfolio P&L Comparison  (Initial=1,000,000 KRW)", fontsize=12)
    ax.set_xlabel("Time Step (Val Set)"); ax.set_ylabel("Portfolio Value (M KRW)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "12th_002_pnl_comparison.png")


def plot_hit_rate_comparison(results, y_val):
    models = list(results.keys())
    colors = ["#2980B9","#27AE60","#8E44AD"]
    x = np.arange(len(QUANTILES)); w = 0.25
    fig, ax = plt.subplots(figsize=(11,5))
    for i,(name,c) in enumerate(zip(models,colors)):
        hr = [results[name]["hit"][t] for t in QUANTILES]
        ax.bar(x+i*w-w, hr, w, label=name, color=c, alpha=0.85)
    ax.plot(x, QUANTILES, "ko--", lw=1.5, ms=7, label="Ideal (= τ)", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([f"τ={t}" for t in QUANTILES], fontsize=10)
    ax.set_title("Quantile Hit Rate Comparison  (Val Set)", fontsize=12)
    ax.set_ylabel("Hit Rate"); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); save_fig(fig, "12th_003_hit_rate_compare.png")


def plot_var_breach_timeline(results, y_val):
    models = list(results.keys())
    colors = ["#2980B9","#27AE60","#8E44AD"]
    fig, axes = plt.subplots(3,1, figsize=(14,10), sharex=True)
    x = np.arange(len(y_val))
    for i,(name,c) in enumerate(zip(models,colors)):
        yp = results[name]["yp"]; var_lo = yp[:,0]
        fail = y_val < var_lo
        axes[i].plot(x, y_val, lw=0.7, color="#A84B2F", alpha=0.7, label="Actual")
        axes[i].plot(x, var_lo, lw=1.2, color=c, ls="--", label=f"VaR τ=0.05")
        axes[i].fill_between(x, var_lo, y_val, where=fail, alpha=0.45, color="#E74C3C", label=f"Failures ({fail.mean()*100:.1f}%)")
        axes[i].set_title(name, fontsize=10); axes[i].legend(fontsize=7, loc="upper right"); axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel("Time Step (Val Set)")
    plt.suptitle("VaR Breach Timeline Comparison", fontsize=12, y=1.01)
    plt.tight_layout(); save_fig(fig, "12th_004_breach_timeline.png")


def plot_backtest_summary_table(results, y_val):
    rows = []
    for name in results:
        r = results[name]
        rows.append([name.split(" (")[0],
                     f"{r['bt']['failure_rate']:.2f}%",
                     f"{r['kupiec']['p_value']:.3f} {'OK' if not r['kupiec']['reject_H0'] else 'FAIL'}",
                     f"{r['christ']['p_value']:.3f} {'OK' if not r['christ']['reject_H0'] else 'FAIL'}",
                     f"{r['crps']:.4f}", f"{r['picp']:.1f}%",
                     f"{r['mpiw']:.4f}"])
    fig, ax = plt.subplots(figsize=(15,3.5))
    ax.axis("off")
    headers=["Model","Fail Rate","Kupiec p","Christoff. p","CRPS","PICP(95%)","MPIW"]
    tbl = ax.table(cellText=rows,colLabels=headers,cellLoc="center",loc="center",colColours=["#2C3E50"]*7)
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.2,2.5)
    for (r,c),cell in tbl.get_celld().items():
        if r==0: cell.set_text_props(color="white",fontweight="bold")
        elif r%2==1: cell.set_facecolor("#EBF5FB")
        else: cell.set_facecolor("#EAFAF1")
    ax.set_title("12차시 VaR 백테스트 종합 결과표", fontsize=12, pad=10)
    plt.tight_layout(); save_fig(fig, "12th_005_summary_table.png")


if __name__ == "__main__":
    results, y_val = run_backtest()
    print("\n  시각화 생성 중...")
    plot_failure_rates(results, y_val)
    plot_pnl_comparison(results, y_val)
    plot_hit_rate_comparison(results, y_val)
    plot_var_breach_timeline(results, y_val)
    plot_backtest_summary_table(results, y_val)
    print("  12차시 완료!")
