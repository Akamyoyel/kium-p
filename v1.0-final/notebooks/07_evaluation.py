"""
notebooks/07_evaluation.py
────────────────────────────────────────────────────────────────────────────
9차시: CRPS·VaR Coverage·Kupiec Test 종합 평가

■ 수행 내용
  1. CRPS (Continuous Ranked Probability Score) 계산
     목표: CRPS ≤ 0.08
  2. VaR(95%) 커버리지 분석
     목표: 실패율 ≈ 5%
  3. Kupiec Test (VaR 모델 통계적 적절성)
     목표: p-value > 0.05 (H₀ 기각 불가)
  4. 3개 모델 비교
     · Baseline (MSE)
     · Multi-Quantile 7차시
     · Optuna 최적화 8차시
  5. 모델별 성능 지표 종합 비교 시각화
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys, warnings, pickle
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config.settings import DATA_SPLIT, FIG_DIR
from src.losses.crps import crps_quantile, crps_per_sample, print_crps_report
from src.evaluation.quantile_metrics import (
    picp, mpiw, winkler_score, var_coverage, kupiec_test,
    quantile_evaluation_report,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

plt.rcParams.update({
    "figure.dpi": 130, "font.family": "DejaVu Sans",
    "axes.unicode_minus": False, "axes.spines.top": False,
    "axes.spines.right": False,
})

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


# ═══════════════════════════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════════════════════════

def load_all_results() -> dict:
    """7차시 · 8차시 결과 로드."""
    d = np.load(DATA_SPLIT / "windows.npz")
    y_val  = d["y_val"].flatten()
    y_test = d["y_test"].flatten()

    with open(MODEL_DIR / "week67_results.pkl", "rb") as f:
        r67 = pickle.load(f)

    optuna_d = torch.load(MODEL_DIR / "optuna_best.pt", map_location="cpu",
                          weights_only=False)

    return {
        "y_val":  y_val,
        "y_test": y_test,
        "w7_yp_val":  r67["r7"]["yp_val"],    # (754, 5)
        "w7_yp_test": r67["r7"]["yp_test"],
        "opt_yp_val":  optuna_d["yp_val"],     # (754, 5)
        "opt_yp_test": optuna_d["yp_test"],
        "opt_params":  optuna_d["best_params"],
    }


# ═══════════════════════════════════════════════════════════════
# 평가 실행
# ═══════════════════════════════════════════════════════════════

def evaluate_all(data: dict) -> dict:
    """3개 모델 전체 평가."""
    results = {}
    for name, yp_val, yp_test in [
        ("Week7 Multi-Q",     data["w7_yp_val"],  data["w7_yp_test"]),
        ("Optuna Best (W8)",  data["opt_yp_val"], data["opt_yp_test"]),
    ]:
        print(f"\n{'─'*55}")
        print(f"  모델: {name}")
        print(f"{'─'*55}")

        # CRPS
        crps_v = print_crps_report(yp_val, data["y_val"], QUANTILES, label=f"{name} Val")

        # 전체 평가 (Val)
        metrics = quantile_evaluation_report(
            yp_val, data["y_val"], QUANTILES,
            crps_score=crps_v, label=f"{name} Val"
        )
        metrics["crps"] = crps_v

        # Test CRPS
        crps_t = crps_quantile(yp_test, data["y_test"], QUANTILES)
        print(f"  Test CRPS: {crps_t:.6f}")

        results[name] = {"metrics": metrics, "crps_val": crps_v, "crps_test": crps_t,
                          "yp_val": yp_val, "yp_test": yp_test}

    return results


# ═══════════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════════

def save_fig(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [Fig] 저장: {p.name}")


def plot_crps_breakdown(results: dict, y_val: np.ndarray) -> None:
    """그림 6: τ별 Pinball Loss + CRPS 분해."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_m = ["#2980B9", "#27AE60"]
    x        = np.arange(len(QUANTILES))
    width    = 0.35

    for i, (name, res) in enumerate(results.items()):
        yp = res["yp_val"]
        pbs = []
        for k, tau in enumerate(QUANTILES):
            e  = y_val - yp[:, k]
            pb = np.where(e >= 0, tau*e, (tau-1)*e).mean()
            pbs.append(pb)

        axes[0].bar(x + i*width - width/2, pbs, width, label=name,
                    color=colors_m[i], alpha=0.85)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"τ={t}" for t in QUANTILES], fontsize=9)
    axes[0].set_title("Per-Quantile Pinball Loss Comparison", fontsize=11)
    axes[0].set_ylabel("Pinball Loss"); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    # CRPS bar
    model_names = list(results.keys())
    crps_vals   = [res["crps_val"] for res in results.values()]
    bars = axes[1].bar(model_names, crps_vals, color=colors_m[:len(model_names)],
                       alpha=0.85, width=0.45)
    axes[1].axhline(0.08, color="red", ls="--", lw=1.8, label="Target (0.08)")
    axes[1].set_title("CRPS Comparison  (Val Set)", fontsize=11)
    axes[1].set_ylabel("CRPS Score"); axes[1].legend(fontsize=9)
    for bar, v in zip(bars, crps_vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                     f"{v:.5f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].grid(alpha=0.3, axis="y")

    plt.suptitle("CRPS & Pinball Loss Analysis (Week 9)", fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig(fig, "9th_001_crps_breakdown.png")


def plot_var_analysis(results: dict, y_val: np.ndarray) -> None:
    """그림 7: VaR(95%) 실패율 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_m = ["#2980B9", "#27AE60"]

    for i, (name, res) in enumerate(results.items()):
        yp   = res["yp_val"]
        var_lower = yp[:, 0]   # τ=0.05
        fail_mask = y_val < var_lower
        fail_rate = fail_mask.mean() * 100

        n = min(300, len(y_val)); x = np.arange(n)
        axes[i].plot(x, y_val[:n], lw=0.8, color="#A84B2F", alpha=0.8, label="Actual")
        axes[i].plot(x, var_lower[:n], lw=1.2, color=colors_m[i],
                     ls="--", label=f"VaR (τ=0.05)")
        axes[i].fill_between(x, var_lower[:n], y_val[:n],
                             where=fail_mask[:n], alpha=0.4, color="#E74C3C",
                             label=f"VaR Failures ({fail_rate:.1f}%)")
        axes[i].set_title(f"{name}\nVaR Failure Rate = {fail_rate:.2f}%  (Target: 5%)",
                          fontsize=10)
        axes[i].set_xlabel("Time Step"); axes[i].set_ylabel("Scaled Log-Return")
        axes[i].legend(fontsize=8); axes[i].grid(alpha=0.3)

    plt.suptitle("VaR(95%) Failure Rate Analysis  (Val Set, first 300 steps)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    save_fig(fig, "9th_002_var_analysis.png")


def plot_kupiec_result(results: dict, y_val: np.ndarray) -> None:
    """그림 8: Kupiec Test 결과 시각화."""
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names, fail_rates, p_vals, target_rates = [], [], [], []

    for name, res in results.items():
        yp = res["yp_val"]
        kup = kupiec_test(y_val, yp[:, 0], alpha=0.05)
        model_names.append(name)
        fail_rates.append(kup["p_hat"])
        p_vals.append(kup["p_value"])

    x = np.arange(len(model_names))
    bars = ax.bar(x, fail_rates, color=["#2980B9","#27AE60"], alpha=0.8, width=0.4,
                  label="Actual Failure Rate")
    ax.axhline(5.0, color="red", ls="--", lw=1.8, label="Target (5.0%)")
    ax.axhline(2.5, color="#E67E22", ls=":", lw=1, alpha=0.6, label="Over-conservative (2.5%)")
    ax.axhline(10.0, color="#E74C3C", ls=":", lw=1, alpha=0.6, label="Under-conservative (10%)")

    for xi, (bar, pv, fr) in enumerate(zip(bars, p_vals, fail_rates)):
        verdict = "✅ H₀ kept" if pv >= 0.05 else "❌ H₀ rejected"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"{fr:.1f}%\np={pv:.3f}\n{verdict}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
    ax.set_title("Kupiec Test — VaR Failure Rate vs Target (5%)", fontsize=12)
    ax.set_ylabel("Failure Rate (%)"); ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    save_fig(fig, "9th_003_kupiec_test.png")


def plot_model_comparison_table(results: dict, y_val: np.ndarray) -> None:
    """그림 9: 전체 모델 성능 비교표 (시각화)."""
    rows = []
    headers = ["Model", "CRPS", "PICP(95%)", "MPIW", "Winkler", "Fail Rate", "Kupiec p"]
    colors_row = [["#ECF0F1"]*7, ["#D5E8D4"]*7]

    for name, res in results.items():
        yp  = res["yp_val"]
        crps_v = res["crps_val"]
        picp_v = picp(y_val, yp[:,0], yp[:,4])
        mpiw_v = mpiw(yp[:,0], yp[:,4])
        wink_v = winkler_score(y_val, yp[:,0], yp[:,4])
        kup_v  = kupiec_test(y_val, yp[:,0])
        rows.append([name, f"{crps_v:.5f}", f"{picp_v:.1f}%",
                     f"{mpiw_v:.5f}", f"{wink_v:.5f}",
                     f"{kup_v['p_hat']:.2f}%", f"{kup_v['p_value']:.4f}"])

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center",
                   loc="center", colColours=["#2C3E50"]*7)
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.3, 2.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#EAFAF1")
        elif r == 2:
            cell.set_facecolor("#EBF5FB")
    ax.set_title("Model Performance Comparison — Week 9 Evaluation", fontsize=12, pad=10)
    plt.tight_layout()
    save_fig(fig, "9th_004_model_comparison.png")


def plot_crps_distribution(results: dict, y_val: np.ndarray) -> None:
    """그림 10: 샘플별 CRPS 분포."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_m = ["#2980B9", "#27AE60"]

    for i, (name, res) in enumerate(results.items()):
        yp     = res["yp_val"]
        crps_s = crps_per_sample(yp, y_val, QUANTILES)
        axes[i].hist(crps_s, bins=40, density=True, alpha=0.75, color=colors_m[i])
        axes[i].axvline(crps_s.mean(), color="red", lw=2, ls="--",
                        label=f"Mean = {crps_s.mean():.5f}")
        axes[i].axvline(0.08, color="#E67E22", lw=1.5, ls=":",
                        label="Target (0.08)")
        axes[i].set_title(f"{name}\nSample-wise CRPS Distribution", fontsize=10)
        axes[i].set_xlabel("CRPS per sample"); axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=8); axes[i].grid(alpha=0.3)

    plt.suptitle("Sample-wise CRPS Distribution  (Val Set)", fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig(fig, "9th_005_crps_distribution.png")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*55}")
    print("  9차시: CRPS·VaR Coverage·Kupiec Test 종합 평가")
    print(f"{'='*55}")

    data    = load_all_results()
    results = evaluate_all(data)

    print("\n  [9차시 시각화 생성]")
    plot_crps_breakdown(results, data["y_val"])
    plot_var_analysis(results, data["y_val"])
    plot_kupiec_result(results, data["y_val"])
    plot_model_comparison_table(results, data["y_val"])
    plot_crps_distribution(results, data["y_val"])

    # 저장
    with open(MODEL_DIR / "week9_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\n  저장: data/models/week9_results.pkl")

    print(f"\n{'='*55}")
    print("  9차시 완료!")
    for name, res in results.items():
        print(f"  [{name}]")
        print(f"    CRPS     = {res['crps_val']:.6f}  (목표≤0.08: {'✅' if res['crps_val']<=0.08 else '❌'})")
        m = res["metrics"]
        print(f"    PICP(95%)= {m['picp_95']:.2f}%  (목표≥90%: {'✅' if m['picp_95']>=90 else '❌'})")
        kup = m["kupiec"]
        print(f"    Kupiec   = p={kup['p_value']:.4f}  ({'✅' if not kup['reject_H0'] else '❌'})")
    print(f"{'='*55}\n")
    return results


if __name__ == "__main__":
    main()
