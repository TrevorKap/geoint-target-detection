# queue_landcover_training.ps1 — wait for the DOTA from-scratch GPU run to
# finish, then automatically train the Sentinel-2 and Landsat impervious-surface
# models (identical U-Net, 100 epochs each) on the now-free GPU. Detached; safe
# to launch and forget.
$ErrorActionPreference = 'Stop'
$root = "C:\Users\Owner\ClaudeProjects\SatelliteObjectDetection"
$py = "$root\.venv\Scripts\python.exe"
$logFile = "$root\ml\outputs\landcover_queue.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Add-Content -Path $logFile -Value $line
}

Log "Watcher started. Waiting for DOTA from-scratch training (PID file: train_scratch.pid) to finish…"
$dotaPid = Get-Content "$root\ml\outputs\train_scratch.pid" -ErrorAction SilentlyContinue
while ($dotaPid -and (Get-Process -Id $dotaPid -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
Log "DOTA training finished (or PID not found). GPU should be free. Starting landcover training."

foreach ($sensor in @("sentinel2", "landsat")) {
    Log "=== Training $sensor (100 epochs, GPU) ==="
    & $py "$root\ml\src\train_landcover.py" --sensor $sensor --epochs 100 --batch 16 --device 0 *>> $logFile
    Log "=== $sensor done ==="
}

Log "Both landcover models trained. Ready to add to the model registry."
