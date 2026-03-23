import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- 설정 ---
# yfinance와 pandas 라이브러리가 설치되어 있지 않다면, 터미널에서 아래 명령어를 실행하세요.
# pip install yfinance pandas

# 다운로드할 티커 설정
# 예: 삼성전자 (005930.KS), S&P 500 (^GSPC), VIX (^VIX), 원/달러 환율 (USDKRW=X)
TICKERS = ["005930.KS", "^GSPC", "^VIX", "USDKRW=X"]

# 데이터 다운로드 기간 (최근 1년)
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365)

# 데이터 간격 (5분)
# 참고: yfinance는 5분봉 데이터에 대해 최근 60일치만 제공하는 제한이 있을 수 있습니다.
# 더 긴 기간의 데이터를 원할 경우, 데이터 제공 업체나 다른 API를 고려해야 할 수 있습니다.
INTERVAL = "5m"


def build_daily_windows(start_date, end_date):
    """[start, end) 형태의 1일 구간 리스트를 생성합니다."""
    windows = []
    cursor = start_date
    while cursor < end_date:
        next_day = min(cursor + timedelta(days=1), end_date)
        windows.append((cursor, next_day))
        cursor = next_day
    return windows


def download_intraday_by_day(ticker, start_date, end_date, interval):
    """5분봉 데이터를 하루 단위로 나눠 요청한 뒤 병합합니다."""
    frames = []
    failed_days = 0

    for day_start, day_end in build_daily_windows(start_date, end_date):
        try:
            chunk = yf.download(
                ticker,
                start=day_start,
                end=day_end,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if chunk.empty:
                failed_days += 1
                continue
            frames.append(chunk)
        except Exception:
            failed_days += 1

    if not frames:
        return pd.DataFrame(), failed_days

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df, failed_days

# --- 데이터 다운로드 ---
def download_financial_data(tickers, start_date, end_date, interval):
    """
    지정된 티커 목록에 대해 금융 데이터를 다운로드합니다.
    """
    data = {}
    for ticker in tickers:
        try:
            print(f"{ticker} 데이터 다운로드 중...")
            # 5분봉을 1일 단위로 분할 요청합니다.
            df, failed_days = download_intraday_by_day(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
            if df.empty:
                print(f"{ticker}에 대한 데이터를 찾을 수 없습니다.")
            else:
                # 거래량(Volume)이 없는 경우 (예: 환율, 지수) 해당 열을 0으로 채웁니다.
                if 'Volume' not in df.columns:
                    df['Volume'] = 0
                data[ticker] = df
                print(f"{ticker} 다운로드 완료. 행 수={len(df)}, 실패 일수={failed_days}")
        except Exception as e:
            print(f"{ticker} 다운로드 중 오류 발생: {e}")
    return data

# --- 데이터 검증 및 샘플링 ---
def validate_and_sample_data(data):
    """
    다운로드한 데이터를 검증하고 기본 정보를 출력합니다.
    """
    for ticker, df in data.items():
        print(f"\n--- {ticker} 데이터 검증 ---")
        print("데이터 샘플 (상위 5개):")
        print(df.head())
        print("\n기본 정보:")
        df.info()
        print("\n결측치 확인:")
        print(df.isnull().sum())

if __name__ == "__main__":
    # 1. 데이터 다운로드
    financial_data = download_financial_data(TICKERS, START_DATE, END_DATE, INTERVAL)

    # 2. 데이터 검증
    if financial_data:
        validate_and_sample_data(financial_data)
        # 데이터를 파일로 저장 (예: CSV)
        for ticker, df in financial_data.items():
            safe_ticker_name = ticker.replace('^', '').replace('=', '_')
            df.to_csv(f"{safe_ticker_name}_{INTERVAL}_data.csv")
            print(f"\n{ticker} 데이터를 {safe_ticker_name}_{INTERVAL}_data.csv 파일로 저장했습니다.")
    else:
        print("다운로드된 데이터가 없습니다.")
