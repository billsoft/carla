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
#   firewall Allow inbound TCP 2000-2002 (RPC/stream/secondary) and 8000 (TM)
#
# Auth: OpenSSH public key (id_ed25519). No passwords stored anywhere.
# Remote layout:
#   D:\carla-server\Windows\CarlaUnreal.exe        (package contents)
#   D:\carla-update\carla-development.zip          (uploaded zip)
#   D:\carla-server.bak                            (previous version)
# =====================================================================
param(
    [ValidateSet("status", "start", "stop", "restart", "logs", "deploy", "test", "firewall", "viewer")]
    [string]$Action = "status",
    [string]$RemoteHost = "192.168.100.81",
    [string]$RemoteUser = "Administrator",
    [string]$RemoteRoot = "D:\carla-server",
    [string]$TaskName = "CARLA-Server",
    # UE5.8 package (this machine junctions Build\Package to C:\carla_build58\Package)
    [string]$ZipPath = "D:\code\carla_ue58\Build\Package\Carla-0.10.0-Win64-Development.zip",
    [string]$ClientPython = "C:\Users\bills\.conda\envs\carla58\python.exe",
    # Realism defaults (2026-10-14, measured on 81: no change to the 16-camera tick time, ~1.25 s):
    #   Lumen GI/reflections on hardware RT cores (more accurate glass/facade reflections) and a larger RT
    #   geometry pool (the 2 GB default overflowed: "RAY TRACING GEOMETRY - REQUESTED MEMORY OVER BUDGET").
    #   r.RayTracing.Shadows stays off: RT does not support WPO foliage, so tree/grass shadows would be wrong.
    [string]$CarlaArgs = "-carla-server -quality-level=Epic -RenderOffScreen -nosound -ini:Engine:[/Script/Engine.RendererSettings]:r.Lumen.HardwareRayTracing=True,[SystemSettings]:r.RayTracing.ResidentGeometryMemoryPoolSizeInMB=8000",
    [int]$Tail = 50,
    # deploy: reuse the zip already uploaded to D:\carla-update (retry after a failure past the upload)
    [switch]$SkipUpload
)

$ErrorActionPreference = "Stop"
$sshTarget = "$RemoteUser@$RemoteHost"

function Invoke-RemotePs([string]$Script) {
    # Base64(UTF-16LE) EncodedCommand: immune to quoting/newline/encoding issues.
    # 2>$null: remote error records arrive as CliXml over stderr and fail to
    # re-parse locally; the remote scripts report status as stdout strings.
    $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Script))
    # Windows PowerShell 5.1 turns ANY stderr line of a native exe into a terminating
    # NativeCommandError under ErrorActionPreference=Stop (even with 2>$null), e.g. the
    # harmless "task not found" from schtasks /End on first deploy. Judge by exit code only.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $out = & ssh.exe -o BatchMode=yes -o ConnectTimeout=15 $sshTarget "powershell -NoProfile -EncodedCommand $enc" 2>$null
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
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
# schtasks /TR is limited to 261 chars, which long -ini:/cvar overrides exceed (the /Create then fails and
# /Run silently starts the previously registered command). The task runs a launcher .bat instead; it lives
# outside $RemoteRoot so deploy (which swaps that folder) does not delete it.
New-Item -ItemType Directory -Force -Path 'D:\carla-launch' | Out-Null
`$lines = @('@echo off', 'cd /d "$exeDir"', '"CarlaUnreal.exe" $CarlaArgs')
`$nl = [string][char]13 + [string][char]10
[IO.File]::WriteAllText('D:\carla-launch\start_carla.bat', ((`$lines -join `$nl) + `$nl), [Text.Encoding]::ASCII)
Write-Output ('launcher: ' + (Get-Content 'D:\carla-launch\start_carla.bat' | Select-Object -Last 1))
# ONSTART: the server comes back by itself after the machine reboots (always-on server)
schtasks.exe /Create /TN $TaskName /TR 'D:\carla-launch\start_carla.bat' /SC ONSTART /RU SYSTEM /RL HIGHEST /F | Out-Null
if(`$LASTEXITCODE -ne 0){ Write-Output 'ERROR: schtasks /Create failed'; exit 1 }
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
cmd.exe /c "schtasks /End /TN $TaskName >nul 2>&1"
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

        if ($SkipUpload) {
            Write-Host "[2/4] -SkipUpload: reusing D:\carla-update\carla-development.zip on the server"
        } else {
            Write-Host "[2/4] Uploading via scp (gigabit ~20min for 15GB)..."
            & scp.exe -o BatchMode=yes $ZipPath "${sshTarget}:D:/carla-update/carla-development.zip"
            if ($LASTEXITCODE -ne 0) { throw "scp upload failed" }
        }

        Write-Host "[3/4] Stopping server, extracting, swapping versions..."
        $script = @"
`$ErrorActionPreference='Stop'
cmd.exe /c "schtasks /End /TN $TaskName >nul 2>&1"
Stop-Process -Name CarlaUnreal -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
if(Test-Path 'D:\carla-update\extracted'){ Remove-Item 'D:\carla-update\extracted' -Recurse -Force }
Write-Output 'Extracting with tar.exe (much faster than Expand-Archive on a 10GB+ zip)...'
New-Item -ItemType Directory -Force -Path 'D:\carla-update\extracted' | Out-Null
tar.exe -xf 'D:\carla-update\carla-development.zip' -C 'D:\carla-update\extracted'
if(`$LASTEXITCODE -ne 0){ throw "tar extract failed (`$LASTEXITCODE)" }
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

    "viewer" {
        # pygame preview window on the server's own desktop (remote\carla_viewer.py, free roam WASD/QE + mouse).
        # The server itself stays headless in Session 0; this is just another read-only client connecting to
        # localhost, launched in the interactive session of the logged-on Administrator (console or RDP) and
        # at every logon. It reconnects by itself while the server starts/restarts. Needs the carla58 conda env
        # (C:\ProgramData\miniconda3\envs\carla58, python 3.10 + pygame + the carla wheel) on the server.
        Assert-KeyAuth
        Write-Host "== Uploading viewer and registering task 'CARLA-Viewer' (interactive, at logon) ==" -ForegroundColor Cyan
        & scp.exe -o BatchMode=yes "$PSScriptRoot\carla_viewer.py" "${sshTarget}:D:/carla-launch/carla_viewer.py"
        if ($LASTEXITCODE -ne 0) { throw "scp viewer failed" }
        $script = @"
`$pyw = 'C:\ProgramData\miniconda3\envs\carla58\pythonw.exe'
if(-not (Test-Path `$pyw)){ Write-Output 'ERROR: carla58 env missing on server (see carla_remote.ps1 -Action viewer)'; exit 1 }
`$action = New-ScheduledTaskAction -Execute `$pyw -Argument 'D:\carla-launch\carla_viewer.py --host localhost --render-width 1280 --render-height 720' -WorkingDirectory 'D:\carla-launch'
`$trigger = New-ScheduledTaskTrigger -AtLogOn -User '$RemoteUser'
`$principal = New-ScheduledTaskPrincipal -UserId '$RemoteUser' -LogonType Interactive -RunLevel Limited
`$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'CARLA-Viewer' -Action `$action -Trigger `$trigger -Principal `$principal -Settings `$settings -Force | Out-Null
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" | Where-Object { `$_.CommandLine -match 'carla_viewer' } | ForEach-Object { Stop-Process -Id `$_.ProcessId -Force }
# Let Task Scheduler notice the killed instance first, or MultipleInstances=IgnoreNew drops this start.
Start-Sleep -Seconds 3
Start-ScheduledTask -TaskName 'CARLA-Viewer'
Start-Sleep -Seconds 5
`$p = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" | Where-Object { `$_.CommandLine -match 'carla_viewer' }
if(`$p){ Write-Output ('viewer running pid=' + `$p.ProcessId + ' session=' + `$p.SessionId) } else { Write-Output 'WARN: viewer not running (is Administrator logged on, console or RDP?)' }
"@
        Invoke-RemotePs $script
    }

    "firewall" {
        Assert-KeyAuth
        Write-Host "== Opening inbound TCP 2000-2002, 8000 on $RemoteHost ==" -ForegroundColor Cyan
        $script = @"
if(-not (Get-NetFirewallRule -DisplayName 'CARLA Server' -ErrorAction SilentlyContinue)){
  New-NetFirewallRule -DisplayName 'CARLA Server' -Direction Inbound -Protocol TCP -LocalPort 2000-2002,8000 -Action Allow -Profile Any | Out-Null
  Write-Output 'rule created'
} else { Write-Output 'rule already exists' }
Get-NetFirewallRule -DisplayName 'CARLA Server' | Get-NetFirewallPortFilter | ForEach-Object { 'ports: ' + (`$_.LocalPort -join ',') }
"@
        Invoke-RemotePs $script
    }

    "test" {
        Write-Host "== Local RPC test against ${RemoteHost}:2000 ==" -ForegroundColor Cyan
        $py = $ClientPython
        if (-not (Test-Path $py)) { throw "Python not found: $py" }
        $code = "import carla; c=carla.Client('$RemoteHost',2000); c.set_timeout(30.0); print('server version:', c.get_server_version())"
        & $py -c $code
    }
}
