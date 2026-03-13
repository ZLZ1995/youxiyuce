@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo 正在启动程序...
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    py -m streamlit run "game_sales_monthly_pro.py" --server.port 8501 --server.headless true
) else (
    python -m streamlit run "game_sales_monthly_pro.py" --server.port 8501 --server.headless true
)

echo.
echo 程序已退出，请检查上面的报错信息。
pause