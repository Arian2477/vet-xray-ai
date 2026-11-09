# Continue the stopped FULL run from its last saved epoch, with the GPU temperature guard on. The test set stays locked.
# Run: powershell -ExecutionPolicy Bypass -File G:\Vet_XRay_AI\v2\resume_training.ps1
$py = 'G:\Vet_XRay_AI\.venv\Scripts\python.exe'
$here = 'G:\Vet_XRay_AI\v2'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:VXR_MODE = 'FULL'
$env:VXR_FOLDS = '5'
$env:VXR_STOP_BEFORE_TEST = '1'

$guard = Start-Process -FilePath $py -ArgumentList '-I', "$here\gpu_watchdog.py", '--stop', '82', '--resume', '68' -WindowStyle Hidden -PassThru
try {
    & $py -I "$here\xray_finetuning_v2.py" *>> "$here\full_run.log"
    "training exit code: $LASTEXITCODE"
} finally {
    if (-not $guard.HasExited) { Stop-Process -Id $guard.Id -Force }
}
