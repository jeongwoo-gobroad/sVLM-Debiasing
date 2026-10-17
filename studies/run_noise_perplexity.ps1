param(
    [Parameter(Mandatory=$true)][string]$PythonQwen,
    [Parameter(Mandatory=$true)][string]$PythonKanana,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
Set-Location -LiteralPath (Split-Path $PSScriptRoot)
$outputPath = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$lock = [IO.File]::Open((Join-Path $outputPath 'pipeline.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
$state = @{stage='running'; pid=$PID; jobs=@{}; started=[DateTime]::UtcNow.ToString('o')}
function Save-State {
    [IO.File]::WriteAllText((Join-Path $outputPath 'pipeline_status.json'), ($state | ConvertTo-Json -Depth 7), [Text.UTF8Encoding]::new($false))
}
try {
    foreach ($job in @(@('qwen3.5_4b',$PythonQwen), @('kanana',$PythonKanana))) {
        $model = $job[0]
        $state.current = $model; $state.jobs[$model] = 'running'; Save-State
        Write-Host ('Noise-conditioned tag perplexity: ' + $model + ' / 124 candidates')
        & $job[1] -X utf8 (Join-Path $PSScriptRoot 'capture_noise_perplexity.py') --model $model --output $outputPath 2>&1 | Tee-Object -FilePath (Join-Path $outputPath ($model + '.log'))
        if ($LASTEXITCODE -ne 0) { throw ($model + ' failed; inspect status and log.') }
        $state.jobs[$model] = 'complete'; Save-State
    }
    $state.stage = 'complete'; $state.current = $null
    $state.finished = [DateTime]::UtcNow.ToString('o'); Save-State
    Write-Host 'Both models complete. All 248 candidate scores and causal validation records are saved.'
} catch {
    $state.stage = 'failed'; $state.error = $_.ToString()
    if ($state.current) { $state.jobs[$state.current] = 'failed' }
    Save-State; Write-Error $_
} finally { $lock.Dispose() }
