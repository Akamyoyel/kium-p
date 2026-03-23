import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

# --- 설정 ---
# 분석할 데이터 파일 설정
# download_data.py에서 저장된 파일 이름을 사용합니다.
# 예: 삼성전자 (005930_KS_5m_data.csv), VIX (^VIX_5m_data.csv) 등
MAIN_TICKER_FILE = "005930_KS_5m_data.csv"
EXOGENOUS_FILES = {
    "VIX": "^VIX_5m_data.csv",
    "USDKRW": "USDKRW_X_5m_data.csv"
}
INTERVAL = "5m" # 데이터 간격 (파일 이름과 일치)

# --- 데이터 로드 및 전처리 ---
def load_and_preprocess_data(file_path):
    """
    CSV 파일을 로드하고 기본적인 전처리를 수행합니다.
    """
    try:
        df = pd.read_csv(file_path, index_col='Datetime', parse_dates=True)
        
        # 결측치 처리 (forward fill)
        # 시계열 데이터에서는 이전 값으로 채우는 것이 일반적인 방법 중 하나입니다.
        df.ffill(inplace=True)
        
        # 수익률 계산 (로그 수익률 사용)
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 첫 번째 행의 수익률은 NaN이므로 제거
        df.dropna(inplace=True)
        
        return df
    except FileNotFoundError:
        print(f"오류: {file_path} 파일을 찾을 수 없습니다. 먼저 download_data.py를 실행하여 데이터를 다운로드하세요.")
        return None

# --- 탐색적 데이터 분석 (EDA) ---
def perform_eda(main_df, main_ticker_name):
    """
    주요 티커 데이터에 대한 탐색적 분석 및 시각화를 수행합니다.
    """
    print(f"\n--- {main_ticker_name} 탐색적 데이터 분석 ---")

    # 1. 결측치/이상치 확인 (시각화) 및 분포 확인
    plt.figure(figsize=(14, 6))
    sns.histplot(main_df['Returns'], kde=True, bins=100)
    plt.title(f'{main_ticker_name} Returns Distribution ({INTERVAL})')
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # 2. Volatility Clustering (변동성 군집) 검증
    plt.figure(figsize=(14, 6))
    main_df['Returns'].plot()
    plt.title(f'{main_ticker_name} Returns Over Time - Volatility Clustering ({INTERVAL})')
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.grid(True)
    plt.show()
    print("수익률 시계열 그래프에서 변동성이 큰 기간과 작은 기간이 뭉쳐 나타나는 '변동성 군집' 현상을 확인할 수 있습니다.")

    # 3. ACF/PACF 분석
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(main_df['Returns']**2, lags=50, ax=axes[0], title=f'{main_ticker_name} Squared Returns ACF')
    plot_pacf(main_df['Returns']**2, lags=50, ax=axes[1], title=f'{main_ticker_name} Squared Returns PACF')
    plt.show()
    print("수익률의 제곱(변동성의 대리 변수)에 대한 ACF 플롯은 GARCH 모델 적용의 필요성을 시사하는 자기상관성을 보여줍니다.")


# --- 상관관계 분석 ---
def analyze_correlation(main_df, exo_files, main_ticker_name):
    """
    주요 티커와 외생 변수들 간의 상관관계를 분석합니다.
    """
    print("\n--- 상관관계 분석 ---")
    
    # 외생 변수 데이터 로드
    returns_df = pd.DataFrame(main_df['Returns']).rename(columns={'Returns': main_ticker_name})
    
    for name, file in exo_files.items():
        exo_df = load_and_preprocess_data(file)
        if exo_df is not None:
            returns_df[name] = exo_df['Returns']
            
    # 모든 데이터의 인덱스를 기준으로 병합하고 결측치 제거
    returns_df.dropna(inplace=True)

    # 상관관계 히트맵
    plt.figure(figsize=(8, 6))
    correlation_matrix = returns_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Returns')
    plt.show()
    print("상관관계 히트맵은 각 자산 수익률 간의 선형 관계를 보여줍니다.")


if __name__ == "__main__":
    # 1. 메인 데이터 로드 및 분석
    main_ticker_name = MAIN_TICKER_FILE.split('_')[0]
    main_data = load_and_preprocess_data(MAIN_TICKER_FILE)
    
    if main_data is not None:
        # 2. 탐색적 데이터 분석 수행
        perform_eda(main_data, main_ticker_name)
        
        # 3. 상관관계 분석 수행
        analyze_correlation(main_data, EXOGENOUS_FILES, main_ticker_name)
    else:
        print("분석을 진행할 수 없습니다.")
