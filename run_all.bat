@echo off
echo ==============================================================================
echo Running MDI3003 Lab 07 End-to-End Recommender Pipeline
echo Student: Lokanth S (23MID0037) ^| Faculty: Dr. Durgesh Kumar
echo ==============================================================================

python scripts/run_all.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pipeline execution failed!
    exit /b %ERRORLEVEL%
)

python scripts/make_notebook.py
python scripts/validate_submission.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Validation failed!
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] Full pipeline, validation, and artifacts generated cleanly!
