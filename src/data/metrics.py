"""
src/evaluation/metrics.py
────────────────────────────────────────────────────────────────────────────
5차시: 회귀 평가 지표

· MAE  (Mean Absolute Error)
· RMSE (Root Mean Squared Error)
· MAPE (Mean Absolute Percentage Error)
· R²   (Coefficient of Determination)
· 방향 정확도 (Directional Accuracy)

이 지표들은 6~9차시 Quantile 모델의 Point Prediction Baseline 비교에 사용된다.
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error = mean(|y - ŷ|)"""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error = sqrt(mean((y - ŷ)²))"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error = mean(|y - ŷ| / (|y| + eps)) × 100
    eps: 0-division 방지
    """
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² = 1 - SS_res / SS_tot
    1에 가까울수록 예측력 높음. 음수 = 평균보다 나쁜 예측.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1 - ss_res / ss_tot)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    방향 정확도 = 실제 부호와 예측 부호가 일치하는 비율 (%)
    금융 수익률 예측에서 중요한 실용 지표.
    """
    correct = np.sign(y_true) == np.sign(y_pred)
    return float(np.mean(correct) * 100)


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label:  str = "Model",
) -> dict:
    """
    전체 회귀 지표를 딕셔너리로 반환하고 콘솔에 출력.

    Returns
    -------
    dict  {mae, rmse, mape, r2, dir_acc}
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    result = {
        "mae":     mae(y_true, y_pred),
        "rmse":    rmse(y_true, y_pred),
        "mape":    mape(y_true, y_pred),
        "r2":      r2_score(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
    }

    print(f"\n{'═'*48}")
    print(f"  {label}  —  Regression Metrics")
    print(f"{'═'*48}")
    print(f"  MAE               : {result['mae']:.6f}")
    print(f"  RMSE              : {result['rmse']:.6f}")
    print(f"  MAPE              : {result['mape']:.4f}%")
    print(f"  R²                : {result['r2']:.6f}")
    print(f"  Directional Acc.  : {result['dir_acc']:.2f}%")
    print(f"{'═'*48}\n")

    return result
