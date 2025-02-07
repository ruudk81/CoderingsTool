@echo off
cd /d "C:\Users\rkn\Open survey responses\qualitative_analysis_app"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& ..\textenv\Scripts\Activate.ps1; python -m streamlit run app.py --server.fileWatcherType none"
pause