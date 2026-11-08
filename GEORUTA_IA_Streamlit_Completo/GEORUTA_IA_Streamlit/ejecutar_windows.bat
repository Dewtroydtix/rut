@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 (
    echo Se necesita Python 3.10 o superior instalado.
    echo Consulta README.md para las instrucciones.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Preparando el entorno de Python...
    py -m venv .venv
    if errorlevel 1 goto error
)
echo [2/3] Instalando bibliotecas...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
echo [3/3] Iniciando GEORUTA-IA...
".venv\Scripts\python.exe" -m streamlit run app.py
goto end
:error
echo Ocurrio un error. Consulta README.md
pause
:end
