import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests


def load_dotenv_file(dotenv_path: str = ".env") -> None:
    """간단한 .env 파서: KEY=VALUE 형식을 os.environ에 로드합니다."""
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class KISConfig:
    app_key: str
    app_secret: str
    base_url: str
    tr_id: str
    custtype: str = "P"
    token_retry_count: int = 3
    token_retry_wait_sec: float = 60.0


class KISClient:
    def __init__(self, config: KISConfig):
        self.config = config
        self.access_token = self._issue_access_token()

    def _issue_access_token(self) -> str:
        url = f"{self.config.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
        }
        for attempt in range(1, self.config.token_retry_count + 1):
            response = requests.post(url, json=body, timeout=20)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                body_text = response.text
                error_code = ""
                try:
                    error_code = response.json().get("error_code", "")
                except Exception:
                    pass

                is_rate_limited = error_code == "EGW00133" or "1분당 1회" in body_text
                if is_rate_limited and attempt < self.config.token_retry_count:
                    print(
                        f"토큰 발급 제한(EGW00133): {self.config.token_retry_wait_sec:.0f}초 대기 후 재시도 "
                        f"({attempt}/{self.config.token_retry_count})"
                    )
                    time.sleep(self.config.token_retry_wait_sec)
                    continue

                raise RuntimeError(
                    "토큰 발급 HTTP 오류: "
                    f"status={response.status_code}, url={url}, body={body_text}"
                ) from exc

            payload = response.json()
            token = payload.get("access_token")
            if token:
                return token

            error_code = payload.get("error_code", "")
            if error_code == "EGW00133" and attempt < self.config.token_retry_count:
                print(
                    f"토큰 발급 제한(EGW00133): {self.config.token_retry_wait_sec:.0f}초 대기 후 재시도 "
                    f"({attempt}/{self.config.token_retry_count})"
                )
                time.sleep(self.config.token_retry_wait_sec)
                continue

            raise RuntimeError(f"토큰 발급 실패: {payload}")

        raise RuntimeError("토큰 발급 재시도 횟수를 초과했습니다.")

    def _headers(self) -> Dict[str, str]:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
            "tr_id": self.config.tr_id,
            "custtype": self.config.custtype,
        }

    def inquire_time_itemchartprice(
        self,
        stock_code: str,
        input_hour: str,
        include_past: str = "Y",
        market_div_code: str = "J",
        etc_cls_code: str = "",
    ) -> pd.DataFrame:
        """주식당일분봉조회 호출 결과(output2)를 DataFrame으로 반환합니다."""
        endpoint = "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        url = f"{self.config.base_url}{endpoint}"
        params = {
            "FID_ETC_CLS_CODE": etc_cls_code,
            "FID_COND_MRKT_DIV_CODE": market_div_code,
            "FID_INPUT_ISCD": stock_code,
            "FID_INPUT_HOUR_1": input_hour,
            "FID_PW_DATA_INCU_YN": include_past,
        }
        response = requests.get(url, headers=self._headers(), params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()

        if payload.get("rt_cd") != "0":
            msg = payload.get("msg1", "알 수 없는 오류")
            raise RuntimeError(f"{stock_code} 조회 실패: {msg}")

        rows = payload.get("output2", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df


def parse_kis_minute_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "stck_bsop_date" not in df.columns or "stck_cntg_hour" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["datetime"] = pd.to_datetime(
        work["stck_bsop_date"].astype(str) + work["stck_cntg_hour"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )

    numeric_candidates = [
        "stck_oprc",
        "stck_hgpr",
        "stck_lwpr",
        "stck_prpr",
        "cntg_vol",
        "acml_tr_pbmn",
    ]
    for col in numeric_candidates:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["datetime"]).sort_values("datetime")
    work = work.drop_duplicates(subset=["datetime"], keep="last")
    return work


def to_ohlcv_5m(df: pd.DataFrame) -> pd.DataFrame:
    """1분(또는 불규칙) 체결 데이터를 5분 OHLCV로 집계합니다."""
    if df.empty:
        return df

    # 국내 정규장(09:00~15:30) 데이터만 사용
    df = df[(df["datetime"].dt.hour > 9) | ((df["datetime"].dt.hour == 9) & (df["datetime"].dt.minute >= 0))]
    df = df[(df["datetime"].dt.hour < 15) | ((df["datetime"].dt.hour == 15) & (df["datetime"].dt.minute <= 30))]
    if df.empty:
        return pd.DataFrame()

    idx = df.set_index("datetime")
    has_ohlc = all(col in idx.columns for col in ["stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr"])

    if has_ohlc:
        ohlcv = pd.DataFrame(
            {
                "Open": idx["stck_oprc"].resample("5min").first(),
                "High": idx["stck_hgpr"].resample("5min").max(),
                "Low": idx["stck_lwpr"].resample("5min").min(),
                "Close": idx["stck_prpr"].resample("5min").last(),
                "Volume": idx["cntg_vol"].resample("5min").sum() if "cntg_vol" in idx.columns else 0,
            }
        )
    else:
        # 필수 컬럼이 없으면 현재가 기반으로 최소 집계를 수행
        ohlcv = pd.DataFrame(
            {
                "Open": idx["stck_prpr"].resample("5min").first(),
                "High": idx["stck_prpr"].resample("5min").max(),
                "Low": idx["stck_prpr"].resample("5min").min(),
                "Close": idx["stck_prpr"].resample("5min").last(),
                "Volume": idx["cntg_vol"].resample("5min").sum() if "cntg_vol" in idx.columns else 0,
            }
        )

    ohlcv = ohlcv.dropna(subset=["Open", "High", "Low", "Close"])
    return ohlcv


def fetch_year_5m_with_loop(
    client: KISClient,
    stock_code: str,
    start_dt: datetime,
    end_dt: datetime,
    max_calls: int,
    sleep_sec: float,
    start_base_time: str,
) -> Tuple[pd.DataFrame, int]:
    """반복요청 루프로 시작시각을 뒤로 이동하며 가능한 한 과거 데이터까지 누적합니다."""
    unique_rows: Dict[Tuple[str, str], Dict[str, object]] = {}
    failed_calls = 0
    current_hour = str(start_base_time).zfill(6)
    seen_base_times: set[str] = set()
    best_oldest: datetime | None = None
    target_date = start_dt.strftime("%Y%m%d")

    for call_idx in range(max_calls):
        try:
            raw_df = client.inquire_time_itemchartprice(stock_code=stock_code, input_hour=current_hour, include_past="Y")
            parsed_df = parse_kis_minute_rows(raw_df)
            if parsed_df.empty:
                failed_calls += 1
                break

            # (영업일자, 체결시간) 기준으로 중복 제거
            for row in raw_df.to_dict("records"):
                date_key = str(row.get("stck_bsop_date", "")).strip()
                time_key = str(row.get("stck_cntg_hour", "")).zfill(6)
                if not date_key or not time_key:
                    continue
                unique_rows[(date_key, time_key)] = row

            oldest = parsed_df["datetime"].min().to_pydatetime()
            newest = parsed_df["datetime"].max().to_pydatetime()
            print(
                f"  - {call_idx + 1}회차: rows={len(parsed_df)}, "
                f"window={oldest.strftime('%Y-%m-%d %H:%M:%S')}~{newest.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            if best_oldest is not None and oldest >= best_oldest:
                break
            best_oldest = oldest

            if oldest <= start_dt:
                break

            # 날짜 조건 종료
            oldest_date = oldest.strftime("%Y%m%d")
            if oldest_date <= target_date:
                break

            # 응답 정렬 방향과 무관하게, 가장 과거 시점을 기준으로 다음 호출 진행
            oldest_row = parsed_df.iloc[0]
            next_base_time = oldest_row["datetime"].strftime("%H%M%S")
            if not next_base_time or next_base_time == "000000":
                break

            # 같은 기준시간이 반복되면 더 이상 과거로 전진하지 못하는 상태이므로 종료
            if next_base_time in seen_base_times:
                break

            seen_base_times.add(next_base_time)
            current_hour = next_base_time
            time.sleep(sleep_sec)
        except Exception:
            failed_calls += 1
            time.sleep(sleep_sec)

    if not unique_rows:
        return pd.DataFrame(), failed_calls

    merged_raw = pd.DataFrame(list(unique_rows.values()))
    merged = parse_kis_minute_rows(merged_raw)
    if merged.empty:
        return pd.DataFrame(), failed_calls

    merged = merged[(merged["datetime"] >= start_dt) & (merged["datetime"] <= end_dt)]

    return to_ohlcv_5m(merged), failed_calls


def validate_frame(ticker: str, df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {
            "ticker": ticker,
            "rows": 0,
            "missing_ratio": 1.0,
            "duplicate_rows": 0,
            "start": None,
            "end": None,
        }

    return {
        "ticker": ticker,
        "rows": int(len(df)),
        "missing_ratio": float(df.isna().mean().mean()),
        "duplicate_rows": int(df.index.duplicated().sum()),
        "start": df.index.min(),
        "end": df.index.max(),
    }


def merge_with_existing(out_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """기존 CSV가 있으면 인덱스(datetime) 기준으로 병합 후 중복 제거합니다."""
    if not out_path.exists():
        return new_df.sort_index()

    try:
        existing = pd.read_csv(out_path, index_col=0, parse_dates=True)
    except Exception:
        return new_df.sort_index()

    if existing.empty:
        return new_df.sort_index()

    if new_df.empty:
        return existing.sort_index()

    merged = pd.concat([existing, new_df], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def run(args: argparse.Namespace) -> None:
    load_dotenv_file(args.dotenv)

    app_key = os.getenv("KIS_APP_KEY", "")
    app_secret = os.getenv("KIS_APP_SECRET", "")
    base_url = os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443")
    tr_id = os.getenv("KIS_TR_ID", "FHKST03010200")

    if not app_key or not app_secret:
        raise RuntimeError("환경변수 KIS_APP_KEY, KIS_APP_SECRET를 설정하세요.")

    client = KISClient(
        KISConfig(
            app_key=app_key,
            app_secret=app_secret,
            base_url=base_url,
            tr_id=tr_id,
            token_retry_count=args.token_retry_count,
            token_retry_wait_sec=args.token_retry_wait_sec,
        )
    )

    if args.target_date:
        try:
            target_day = datetime.strptime(args.target_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("--target-date 형식은 YYYY-MM-DD 이어야 합니다.") from exc

        start_dt = target_day.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = target_day.replace(hour=23, minute=59, second=59, microsecond=0)
    else:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=args.days)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows: List[Dict[str, object]] = []

    for ticker in args.tickers:
        print(f"\n[{ticker}] KIS 5분봉 반복요청 시작")
        df_5m, failed = fetch_year_5m_with_loop(
            client=client,
            stock_code=ticker,
            start_dt=start_dt,
            end_dt=end_dt,
            max_calls=args.max_calls,
            sleep_sec=args.sleep_sec,
            start_base_time=args.start_base_time,
        )

        out_path = output_dir / f"{ticker}_KIS_5m.csv"
        had_existing_file = out_path.exists()
        final_df = df_5m
        if args.append_existing:
            final_df = merge_with_existing(out_path, df_5m)

        if not final_df.empty:
            final_df.to_csv(out_path)
            if args.append_existing and had_existing_file:
                print(
                    f"저장 완료(누적): {out_path} "
                    f"(new_rows={len(df_5m)}, total_rows={len(final_df)}, failed_calls={failed})"
                )
            else:
                print(f"저장 완료: {out_path} (rows={len(final_df)}, failed_calls={failed})")
        else:
            print(f"데이터 없음: {ticker} (failed_calls={failed})")

        validation_rows.append(validate_frame(ticker, final_df))

        # API 제약으로 목표기간 미도달 여부 안내
        if not final_df.empty and final_df.index.min().to_pydatetime() > start_dt:
            print(
                "주의: 반복호출을 수행했지만 요청 시작일에 도달하지 못했습니다. "
                "국내 주식 분봉 API의 과거 조회 제약으로 인해 1년 커버리지가 제한될 수 있습니다."
            )

    summary_path = Path(args.validation_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(summary_path, index=False)
    print(f"\n검증 요약 저장: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KIS Developers API 반복요청 루프로 국내주식 5분봉 데이터를 수집합니다.")
    parser.add_argument("--dotenv", default=".env", help="환경변수 파일 경로 (.env)")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["005930", "000660", "035420"],
        help="국내주식 6자리 종목코드 목록",
    )
    parser.add_argument("--days", type=int, default=365, help="요청 목표 기간(일)")
    parser.add_argument(
        "--target-date",
        default=None,
        help="특정 일자만 조회 (YYYY-MM-DD). 지정 시 --days 대신 해당 일자 1일 범위만 사용",
    )
    parser.add_argument("--max-calls", type=int, default=300, help="티커당 최대 반복 호출 횟수")
    parser.add_argument("--sleep-sec", type=float, default=60.0, help="호출 간 대기 시간(초)")
    parser.add_argument("--start-base-time", default="153000", help="최초 조회 기준 시각(HHMMSS)")
    parser.add_argument("--token-retry-count", type=int, default=3, help="토큰 발급 재시도 횟수")
    parser.add_argument("--token-retry-wait-sec", type=float, default=60.0, help="토큰 제한 시 재시도 대기 시간(초)")
    parser.add_argument("--output-dir", default="data/raw", help="CSV 저장 폴더")
    parser.add_argument("--validation-path", default="data/validation_summary_kis.csv", help="검증 요약 CSV 경로")
    parser.add_argument(
        "--append-existing",
        dest="append_existing",
        action="store_true",
        help="기존 CSV가 있으면 누적 병합(중복 datetime 제거) 저장",
    )
    parser.add_argument(
        "--no-append-existing",
        dest="append_existing",
        action="store_false",
        help="기존 CSV 누적 병합 없이 이번 실행 결과만 저장",
    )
    parser.set_defaults(append_existing=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
