param(
    [ValidateSet("raw_llm", "raw_slm", "agent", "rag", "auto", "fast", "full", "Fast & Private Mode", "Full Intelligence Mode")]
    [string]$Mode = "agent",
    [ValidateSet("cli", "streamlit")]
    [string]$Interface = "cli",
    [string]$Message,
    [string]$OllamaUrl = "http://localhost:11434",
    [switch]$SkipOllama,
    [int]$StartupTimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param([string]$ProjectRoot)

    $candidates = @(
        (Join-Path $ProjectRoot ".venv312\Scripts\python.exe"),
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }

    throw "Python was not found. Create .venv312/.venv or install Python first."
}

function Resolve-OllamaExe {
    $ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaCommand) {
        return $ollamaCommand.Source
    }

    $commonPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path -LiteralPath $commonPath) {
        return $commonPath
    }

    throw "Ollama was not found on PATH. Install Ollama or add it to PATH first."
}

function Test-OllamaReady {
    param([string]$BaseUrl)

    try {
        Invoke-RestMethod -Uri "$($BaseUrl.TrimEnd('/'))/api/tags" -Method Get -TimeoutSec 5 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Get-InstalledOllamaModels {
    param([string]$BaseUrl)

    try {
        $response = Invoke-RestMethod -Uri "$($BaseUrl.TrimEnd('/'))/api/tags" -Method Get -TimeoutSec 5
        return @($response.models | ForEach-Object { $_.name })
    } catch {
        return @()
    }
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Resolve-PythonExe -ProjectRoot $projectRoot
$mainScript = Join-Path $projectRoot "main.py"
$streamlitScript = Join-Path $projectRoot "streamlit_app.py"

Write-Host "Launching Smart Customer Support Simulator"
Write-Host "Project root: $projectRoot"
Write-Host "Using Python: $pythonExe"
Write-Host "Interface: $Interface"
Write-Host "Mode: $Mode"

if (-not $SkipOllama) {
    $ollamaExe = Resolve-OllamaExe
    if (Test-OllamaReady -BaseUrl $OllamaUrl) {
        Write-Host "Ollama is already running at $OllamaUrl"
    } else {
        Write-Host "Starting Ollama server..."
        Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WorkingDirectory $projectRoot -WindowStyle Minimized | Out-Null

        $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds 500
            if (Test-OllamaReady -BaseUrl $OllamaUrl) {
                break
            }
        }

        if (-not (Test-OllamaReady -BaseUrl $OllamaUrl)) {
            throw "Ollama did not become ready at $OllamaUrl within $StartupTimeoutSeconds seconds."
        }

        Write-Host "Ollama is ready at $OllamaUrl"
    }

    $requiredModels = @("llama3.1:8b", "phi3:3.8b")
    $installedModels = Get-InstalledOllamaModels -BaseUrl $OllamaUrl
    $missingModels = @($requiredModels | Where-Object { $_ -notin $installedModels })
    if ($missingModels.Count -gt 0) {
        Write-Warning "Missing Ollama models: $($missingModels -join ', '). Pull them with: ollama pull <model>"
    }
}

$env:ENABLE_OLLAMA = if ($SkipOllama) { "0" } else { "1" }
$env:OLLAMA_URL = $OllamaUrl

Push-Location $projectRoot
try {
    if ($Interface -eq "streamlit") {
        if ($PSBoundParameters.ContainsKey("Message") -and -not [string]::IsNullOrWhiteSpace($Message)) {
            Write-Warning "-Message is ignored when -Interface streamlit is used. Use the chat box in the browser instead."
        }
        Write-Host "Launching Streamlit chat UI..."
        & $pythonExe -m streamlit run $streamlitScript
    } else {
        if ($PSBoundParameters.ContainsKey("Message") -and -not [string]::IsNullOrWhiteSpace($Message)) {
            & $pythonExe $mainScript --mode $Mode --message $Message
        } else {
            & $pythonExe $mainScript --mode $Mode
        }
    }
} finally {
    Pop-Location
}
