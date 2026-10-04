@echo off
rem Vuelve a generar visor_arbol.html leyendo el codigo, INDEX.md y conceptos\catalogo.json.
setlocal EnableExtensions

set "PYCMD="
if exist "%~dp0python\python.exe" set PYCMD="%~dp0python\python.exe"
if not defined PYCMD (
  py -3 --version >nul 2>&1 && set "PYCMD=py -3"
)
if not defined PYCMD (
  python --version >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
  echo No encuentro Python. Ejecuta primero el instalador de la carpeta "instaladores".
  pause
  exit /b 1
)

%PYCMD% "%~dp0py\construir_visor.py"
if errorlevel 1 (
  echo.
  echo Error al generar el visor. Revisa el mensaje de arriba.
  pause
  exit /b 1
)
call "%~dp0abrir_arbol.bat"
