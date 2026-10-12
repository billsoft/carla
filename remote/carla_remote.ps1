# =====================================================================
# carla_remote.ps1 - Control the remote CARLA server (192.168.100.81)
#
# Actions:
#   status   Show process / port 2000 / GPU usage on the server
#   start    Register+run the CARLA-Server scheduled task (SYSTEM, S0)
#   stop     Stop the task and kill CarlaUnreal.exe
#   restart  stop + start
#   logs     Tail the newest CarlaUnreal.log on the server
#   deploy   Upload the local Development zip, replace, restart
#   test     Minimal RPC test from THIS machine (needs carla python pkg)
#
# Auth: OpenSSH public key (id_ed25519). No passwords stored anywhere.
# Remote layout:
#   D:\carla-server\Windows\CarlaUnreal.exe        (package contents)
#   D:\carla-update\carla-development.zip          (uploaded zip)
#   D:\carla-server.bak                            (previous version)
# =====================================================================
param(
    [ValidateSet("status", "start", "stop", "restart", "logs", "deploy", "test")]
    [string]$Action = "status",
    [string]$RemoteHost = "192.168.100.81",
    [string]$RemoteUser = "Administrator",
    [string]$RemoteRoot = "D:\carla-server",
    [string]$TaskName = "CARLA-Server",
    [string]$ZipPath = "D:\code\carla\Build\Package\Carla-0.10.0-Win64-Development.zip",
    [string]$CarlaArgs = "-carla-server -quality-level=Epic -RenderOffScreen -nosound",
    [int]$Tail = 50
)

$ErrorActionPreference = "Stop"
$sshTarget = "$RemoteUser@$RemoteHost"

function Invoke-RemotePs([string]$Script) {
    # Base64(UTF-16LE) EncodedCommand: immune to quoting/newline/encoding issues.
    # 2>$null: remote error records arrive as CliXml over stderr and fail to
    # re-parse locally; the remote scripts report status as stdout strings.
    $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Script))
    $out = & ssh.exe -o BatchMode=yes -o ConnectTimeout=15 $sshTarget "powershell -NoProfile -EncodedCommand $enc" 2>$null
    $code = $LASTEXITCODE
    $out | ForEach-Object { Write-Host $_ }
    if ($code -ne 0) { throw "SSH command failed (exit $code)" }
}

function Assert-KeyAuth {
    Write-Host "== Checking key-auth to $sshTarget ==" -ForegroundColor Cyan
    $null = & ssh.exe -o BatchMode=yes -o ConnectTimeout=10 $sshTarget "echo ok" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Key auth failed. Setup: type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh $sshTarget install-key (see SOP)."
    }
    Write-Host "Key auth OK" -ForegroundColor Green
}

switch ($Action) {

    "status" {
        Assert-KeyAuth
        $script = @"
`$ErrorActionPreference='SilentlyContinue'
Write-Output '---- Process ----'
`$p = Get-Process CarlaUnreal
if(`$p){ Write-Output ('CarlaUnreal RUNNING pid=' + (`$p.Id -join ',')) } else { Write-Output 'CarlaUnreal NOT running' }
Write-Output '---- Scheduled task ----'
`$t = schtasks /Query /TN $TaskName /FO LIST 2>`$null
if(`$t){ `$t | Select-String 'Status|TaskName' } else { Write-Output 'task not registered yet' }
Write-Output '---- Port 2000 / 2001 ----'
`$c = Get-NetTCPConnection -LocalPort 2000 -State Listen
if(`$c){ Write-Output 'TCP 2000 LISTENING' } else { Write-Output 'TCP 2000 not listening' }
`$c2 = Get-NetTCPConnection -LocalPort 2001 -State Listen
if(`$c2){ Write-Output 'TCP 2001 LISTENING' } else { Write-Output 'TCP 2001 not listening' }
Write-Output '---- GPU (CarlaUnreal must appear here to prove real GPU render) ----'
nvidia-smi
"@
        Invoke-RemotePs $script
    }

    "start" {
        Assert-KeyAuth
        Write-Host "== Registering and starting task '$TaskName' (SYSTEM, Session 0) ==" -ForegroundColor Cyan
        $exe = "$RemoteRoot\Windows\CarlaUnreal.exe"
        $exeDir = "$RemoteRoot\Windows"
        $script = @"
`$exe = '$exe'
if(-not (Test-Path `$exe)){ Write-Output 'ERROR: CarlaUnreal.exe not found at ' `$exe; exit 1 }
`$tr = 'cmd.exe /c cd /d \"$exeDir\" && \"CarlaUnreal.exe\" $CarlaArgs'
schtasks.exe /Create /TN $TaskName /TR `$tr /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /F | Out-Null
schtasks.exe /Run /TN $TaskName
Write-Output 'Task started. Waiting 15s for the port...'
Start-Sleep -Seconds 15
`$c = Get-NetTCPConnection -LocalPort 2000 -State Listen -ErrorAction SilentlyContinue
if(`$c){ Write-Output 'SUCCESS: TCP 2000 is LISTENING' } else { Write-Output 'WARN: TCP 2000 not listening yet (first cook/shader load can take minutes). Run: carla_remote.ps1 status' }
"@
        Invoke-RemotePs $script
        Write-Host "== Then run: remote_status_carla.bat ; check nvidia-smi shows CarlaUnreal with VRAM ==" -ForegroundColor Yellow
    }

    "stop" {
        Assert-KeyAuth
        Write-Host "== Stopping task and killing CarlaUnreal.exe ==" -ForegroundColor Cyan
        $script = @"
schtasks.exe /End /TN $TaskName 2>`$null
Stop-Process -Name CarlaUnreal -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
if(Get-Process CarlaUnreal -ErrorAction SilentlyContinue){ Write-Output 'STILL RUNNING' } else { Write-Output 'STOPPED' }
"@
        Invoke-RemotePs $script
    }

    "restart" {
        & $PSCommandPath -Action stop -RemoteHost $RemoteHost
        & $PSCommandPath -Action start -RemoteHost $RemoteHost
    }

    "logs" {
        Assert-KeyAuth
        $script = @"
`$log = Get-ChildItem '$RemoteRoot' -Recurse -Filter 'CarlaUnreal.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if(-not `$log){ Write-Output 'No CarlaUnreal.log found under $RemoteRoot'; exit 0 }
Write-Output ('LOG: ' + `$log.FullName)
Get-Content `$log.FullName -Tail $Tail
"@
        Invoke-RemotePs $script
    }

    "deploy" {
        Assert-KeyAuth
        if (-not (Test-Path $ZipPath)) { throw "Zip not found: $ZipPath (run build_package_development.bat first)" }
        $sizeGb = [math]::Round((Get-Item $ZipPath).Length / 1GB, 2)
        Write-Host "== Deploying $ZipPath ($sizeGb GB) ==" -ForegroundColor Cyan

        Write-Host "[1/4] Preparing remote staging dir..."
        Invoke-RemotePs "New-Item -ItemType Directory -Force -Path 'D:\carla-update' | Out-Null; Write-Output 'staging ready'"

        Write-Host "[2/4] Uploading via scp (gigabit ~20min for 15GB)..."
        & scp.exe -o BatchMode=yes $ZipPath "${sshTarget}:D:/carla-update/carla-development.zip"
        if ($LASTEXITCODE -ne 0) { throw "scp upload failed" }

        Write-Host "[3/4] Stopping server, extracting, swapping versions..."
        $script = @"
`$ErrorActionPreference='Stop'
schtasks.exe /End /TN $TaskName 2>`$null
Stop-Process -Name CarlaUnreal -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
if(Test-Path 'D:\carla-update\extracted'){ Remove-Item 'D:\carla-update\extracted' -Recurse -Force }
Write-Output 'Extracting (can take 10-20 min)...'
Expand-Archive -Path 'D:\carla-update\carla-development.zip' -DestinationPath 'D:\carla-update\extracted' -Force
`$inner = Get-ChildItem 'D:\carla-update\extracted' -Directory | Select-Object -First 1
if(-not `$inner){ throw 'extracted package folder not found' }
if(Test-Path '$RemoteRoot.bak'){ Remove-Item '$RemoteRoot.bak' -Recurse -Force }
if(Test-Path '$RemoteRoot'){ Rename-Item '$RemoteRoot' '$RemoteRoot.bak' }
Move-Item `$inner.FullName '$RemoteRoot'
Remove-Item 'D:\carla-update\extracted' -Recurse -Force -ErrorAction SilentlyContinue
Write-Output 'Deployed. New exe:'
Test-Path '$RemoteRoot\Windows\CarlaUnreal.exe'
"@
        Invoke-RemotePs $script

        Write-Host "[4/4] Starting new server..."
        & $PSCommandPath -Action start -RemoteHost $RemoteHost
        Write-Host "== DEPLOY DONE. Rollback path on server: $RemoteRoot.bak ==" -ForegroundColor Green
    }

    "test" {
        Write-Host "== Local RPC test against $RemoteHost:2000 ==" -ForegroundColor Cyan
        $py = "C:\ProgramData\anaconda3\envs\carla\python.exe"
        if (-not (Test-Path $py)) { throw "Python not found: $py" }
        $code = "import carla; c=carla.Client('$RemoteHost',2000); c.set_timeout(30.0); print('server version:', c.get_server_version())"
        & $py -c $code
    }
}
