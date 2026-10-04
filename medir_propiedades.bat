@echo off
rem Mide las propiedades de cada funcion en ESTE ordenador (velocidad, memoria, escalabilidad, alfa real...)
rem y regenera el visor. Tarda unos minutos.
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

%PYCMD% "%~dp0py\medir_propiedades.py"
%PYCMD% "%~dp0py\construir_visor.py"
if errorlevel 1 (
  echo.
  echo Error al generar el visor. Revisa el mensaje de arriba.
  pause
  exit /b 1
)
call "%~dp0abrir_arbol.bat"
