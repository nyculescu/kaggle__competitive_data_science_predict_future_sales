@echo off
Rem Assume python3 and pip are already installed
cd %~dp0
python -m venv ml_env
call 00_activate_ml_env
python -m pip install -U jupyter matplotlib numpy pandas scipy scikit-learn
echo %ml_env created%