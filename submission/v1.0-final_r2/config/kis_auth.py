"""
config/kis_auth.py
KIS Open API 접근 토큰 발급 및 갱신 유틸리티

API 문서 기준:
  - TR_ID  : (인증 불필요, POST /oauth2/tokenP)
  - 유효기간: 개인고객 1일, 법인 3개월
  - 토큰을 .env 또는 로컬 캐시 파일에 저장하여 재발급 최소화
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_URL   = os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443")
APP_KEY    = os.getenv("KIS_APP_KEY",    "")
APP_SECRET = os.getenv("KIS_APP_SECRET", "")

_TOKEN_CACHE = Path(".token_cache.json")   # 프로젝트 루트 임시 저장


def _load_cached_token() -> dict | None:
    """캐시 파일에서 토큰 로드. 만료 시 None 반환."""
    if not _TOKEN_CACHE.exists():
        return None
    data = json.loads(_TOKEN_CACHE.read_text())
    # 만료 1분 전 기준으로 갱신
    if time.time() < data.get("expires_at", 0) - 60:
        return data
    return None


def _save_token_cache(token: str, expires_in: int) -> None:
    data = {
        "access_token": token,
        "expires_at": time.time() + expires_in,
    }
    _TOKEN_CACHE.write_text(json.dumps(data))


def get_access_token(force_refresh: bool = False) -> str:
    """
    유효한 Access Token 반환.
    캐시가 존재하면 재사용, 만료 또는 force_refresh=True 시 재발급.
    """
    if not force_refresh:
        cached = _load_cached_token()
        if cached:
            return cached["access_token"]

    url = f"{BASE_URL}/oauth2/tokenP"
    payload = {
        "grant_type": "client_credentials",
        "appkey":     APP_KEY,
        "appsecret":  APP_SECRET,
    }
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    token      = data["access_token"]
    expires_in = int(data.get("expires_in", 86400))   # 기본 1일
    _save_token_cache(token, expires_in)
    print(f"[kis_auth] 새 토큰 발급 완료 (유효: {expires_in // 3600}시간)")
    return token


def get_headers(token: str) -> dict:
    """표준 REST 요청 헤더 반환."""
    return {
        "content-type":  "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey":        APP_KEY,
        "appsecret":     APP_SECRET,
        "tr_id":         "",          # 각 API 호출 시 덮어씀
        "custtype":      "P",         # P: 개인
    }
