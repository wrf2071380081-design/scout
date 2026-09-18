@echo off
rem scout launcher - pins the Python 3.13 interpreter this project needs.
rem
rem Why this file exists: on this machine `python` resolves to system Python 3.9.7
rem while scout requires >= 3.10, and the project venv is not on PATH.
rem NOTE: keep this file ASCII-only. cmd.exe reads .cmd in the system ANSI
rem codepage (GBK on zh-CN Windows), so non-ASCII comments corrupt parsing.
setlocal
set "SCOUT_HOME=%~dp0"
set "SCOUT_PYTHON=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"

if not exist "%SCOUT_PYTHON%" (
  echo [scout] interpreter not found: %SCOUT_PYTHON%
  echo [scout] fallback: python -m scout.cli ^<command^>  ^(requires Python ^>= 3.10^)
  exit /b 1
)

set "PYTHONPATH=%SCOUT_HOME%src;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"
set "HF_ENDPOINT=https://hf-mirror.com"
"%SCOUT_PYTHON%" -m scout.cli %*
exit /b %ERRORLEVEL%
