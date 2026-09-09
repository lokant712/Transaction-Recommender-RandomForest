Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Running MDI3003 Lab 07 End-to-End Recommender Pipeline" -ForegroundColor Cyan
Write-Host "Student: Lokanth S (23MID0037) | Faculty: Dr. Durgesh Kumar" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

python scripts/run_all.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Pipeline execution failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

python scripts/make_notebook.py
python scripts/validate_submission.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Validation failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[SUCCESS] Full pipeline, validation, and artifacts generated cleanly!" -ForegroundColor Green
