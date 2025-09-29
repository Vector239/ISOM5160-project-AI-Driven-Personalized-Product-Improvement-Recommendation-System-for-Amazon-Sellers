@echo off
REM 安装依赖
pip install torch torchvision torchaudio

REM 运行主程序
python data_cleaning.py

pause