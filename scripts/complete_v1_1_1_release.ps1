# Complete v1.1.1 reproducibility files on GitHub
# Run in PowerShell with Git installed and access to this repository.
$ErrorActionPreference = "Stop"
$package = Join-Path $HOME "Downloads\IMU_v1.1.1_GITHUB_FIX.zip"
if (-not (Test-Path -LiteralPath $package)) {
    throw "Download IMU_v1.1.1_GITHUB_FIX.zip to the Downloads folder first."
}
$temp = Join-Path $env:TEMP ("imu-v111-" + [guid]::NewGuid().ToString("N"))
$clone = Join-Path $temp "repository"
New-Item -ItemType Directory -Path $temp -Force | Out-Null
try {
    git clone https://github.com/squareshorts/sparse-imu-frame-rotation.git $clone
    if ($LASTEXITCODE -ne 0) { throw "Git clone failed." }
    Expand-Archive -LiteralPath $package -DestinationPath $clone -Force
    $a = Join-Path $clone "r2_reanalysis\artifacts\IMU_measurement_extended_analyses.zip"
    $b = Join-Path $clone "r2_reanalysis\artifacts\IMU_relative_angle_calibration_analyses.zip"
    $meta = Join-Path $clone ".zenodo.json"
    foreach ($f in @($a,$b,$meta)) {
        if (-not (Test-Path -LiteralPath $f)) { throw "Required release file missing: $f" }
    }
    if ((Get-Item $a).Length -ne 8457848) { throw "Unexpected extended-analysis ZIP size." }
    if ((Get-Item $b).Length -ne 8662029) { throw "Unexpected relative-angle ZIP size." }
    & git -C $clone add -- .zenodo.json r2_reanalysis/artifacts
    if ($LASTEXITCODE -ne 0) { throw "Git add failed." }
    & git -C $clone commit -m "Complete v1.1.1 executed analysis archive and Zenodo metadata"
    if ($LASTEXITCODE -ne 0) { throw "Git commit failed." }
    & git -C $clone push origin main
    if ($LASTEXITCODE -ne 0) { throw "Git push failed. Repository not updated." }
    Write-Host "SUCCESS: complete release payload is on GitHub main."
    Write-Host "NEXT: Publish a new GitHub release tagged v1.1.1 at the current main commit."
}
finally {
    if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Recurse -Force }
}
