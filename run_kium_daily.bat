@echo off
:: 1. 프로젝트 디렉토리로 이동
cd /d E:\kium

:: 2. D:\ana 폴더의 활성화 스크립트를 직접 호출 (중요)
:: 시스템 Path가 꼬여있어도 강제로 D:\ana의 환경을 사용하게 합니다.
call "D:\ana\Scripts\activate.bat" D:\mini

:: 3. 가상환경 내의 파이썬 실행 파일 경로를 직접 지정 (이중 안전장치)
set PYTHON_EXE=D:\ana\envs\ksi_quant\python.exe
set PYTHONPATH=E:\kium
python -c "from src.data.collector import load_all_tickers; load_all_tickers(collect_today_flag=True)" >> E:\kium\reports\daily_collect.log 2>&1