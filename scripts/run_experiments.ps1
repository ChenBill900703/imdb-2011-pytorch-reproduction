param(
    [string]$RunPrefix = ('repro-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [int]$TimeoutSeconds = 3600,
    [switch]$Evaluate
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($RunPrefix -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Use letters, digits, hyphens or underscores in RunPrefix.' }
if ($TimeoutSeconds -lt 1) { throw 'TimeoutSeconds must be positive.' }
$pythonExe = '.\.venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) { throw 'Run scripts/setup.ps1 first.' }
if (-not (Test-Path 'data/aclImdb/train/unsupBow.feat')) { throw 'Run python -m imdb2011 download first.' }

function Invoke-Recorded([string]$RunId, [string[]]$CommandArgs) {
    & $pythonExe scripts/record_run.py --run-id $RunId --timeout $TimeoutSeconds -- @CommandArgs
    if ($LASTEXITCODE -ne 0) { throw "Run $RunId failed or timed out. Inspect runs/$RunId/stdout.log; evidence is retained." }
}

$classifierRuns = [System.Collections.Generic.List[string]]::new()
foreach ($feature in @('bow', 'delta')) {
    $runId = "$RunPrefix-$feature"
    $outputPath = "runs/$runId/classifier"
    Invoke-Recorded $runId @('-m','imdb2011','fit-svm','--features',$feature,'--output',$outputPath)
    $classifierRuns.Add($outputPath)
}
foreach ($variant in @('semantic', 'full', 'full_unsup')) {
    $runId = "$RunPrefix-$variant"
    $modelPath = "runs/$runId/model"
    Invoke-Recorded $runId @('-m','imdb2011','train','--config',"configs/$variant.json",'--output',$modelPath)
    foreach ($feature in @('vectors','combined')) {
        $classifierId = "$runId-$feature"
        $outputPath = "runs/$classifierId/classifier"
        Invoke-Recorded $classifierId @('-m','imdb2011','fit-svm','--features',$feature,'--checkpoint',"$modelPath/checkpoint.pt",'--output',$outputPath)
        $classifierRuns.Add($outputPath)
    }
}
if ($Evaluate) {
    foreach ($classifierPath in $classifierRuns) {
        $evaluationId = ($classifierPath -split '/')[1] + '-test'
        Invoke-Recorded $evaluationId @('-m','imdb2011','evaluate','--run',$classifierPath)
    }
}
Write-Output "Finished. Outputs use prefix $RunPrefix. Test evaluation requested: $Evaluate"
