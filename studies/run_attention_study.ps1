param(
    [Parameter(Mandatory=$true)][string]$PythonQwen,
    [Parameter(Mandatory=$true)][string]$PythonKanana,
    [Parameter(Mandatory=$true)][string]$Output,
    [Parameter(Mandatory=$true)][string]$FirstTags
)
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$root = Split-Path $PSScriptRoot
Set-Location -LiteralPath $root
$outputPath = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$lockPath = Join-Path $outputPath 'pipeline.lock'
try {
    $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None')
} catch { throw 'An attention pipeline already owns this output directory.' }
$state = @{stage='running'; pid=$PID; started=[DateTime]::UtcNow.ToString('o'); jobs=@{}}
function Save-State {
    $json = $state | ConvertTo-Json -Depth 7
    [IO.File]::WriteAllText((Join-Path $outputPath 'pipeline_status.json'), $json, [Text.UTF8Encoding]::new($false))
}
try {
    foreach ($job in @(@('qwen3.5_4b', $PythonQwen), @('kanana', $PythonKanana))) {
        $model = $job[0]
        $state.current = $model
        $state.jobs[$model] = 'running'
        Save-State
        Write-Host ("First-token attention: " + $model + " / TEST294")
        $arguments = @((Join-Path $PSScriptRoot 'run_attention_study.py'), '--model', $model,
                       '--output', $outputPath, '--first_tags', $FirstTags, '--resume')
        & $job[1] @arguments 2>&1 | Tee-Object -FilePath (Join-Path $outputPath ($model + '.log'))
        if ($LASTEXITCODE -ne 0) { throw ($model + ' failed; inspect its log and status.') }
        $state.jobs[$model] = 'complete'
        Save-State
    }
    $state.stage = 'complete'
    $state.current = $null
    $state.finished = [DateTime]::UtcNow.ToString('o')
    Save-State
    Write-Host 'Both models complete. Raw attention arrays and validation records are saved.'
} catch {
    $state.stage = 'failed'
    $state.error = $_.ToString()
    Save-State
    Write-Error $_
} finally { $lock.Dispose() }
