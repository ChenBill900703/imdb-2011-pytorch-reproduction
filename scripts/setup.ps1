param([string]$Python = "py")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path .venv\Scripts\python.exe)) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12+ is required. Pass -Python with its executable path." }
}
& .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
