# Read-only diagnosis: crash/reboot analysis via Windows event logs
$ErrorActionPreference = 'Continue'

Write-Output '=== Reboot (Kernel-Power 41) timeline, last 6 ==='
Get-WinEvent -FilterHashtable @{LogName='System'; Id=41} -MaxEvents 6 | ForEach-Object {
  '{0}  BugcheckCode={1}' -f $_.TimeCreated, $_.Properties[10].Value
}

Write-Output ''
Write-Output '=== BugCheck (WER 1001) dump records, last 3 ==='
Get-WinEvent -FilterHashtable @{LogName='System'; Id=1001; ProviderName='Microsoft-Windows-WER-SystemErrorReporting'} -MaxEvents 3 -ErrorAction SilentlyContinue | ForEach-Object {
  '{0}  {1}' -f $_.TimeCreated, ($_.Message -replace '\s+', ' ')
}

Write-Output ''
$t = (Get-WinEvent -FilterHashtable @{LogName='System'; Id=41} -MaxEvents 1).TimeCreated
Write-Output ('=== Critical/Error 12 min before last reboot (' + $t + ') ===')
Get-WinEvent -FilterHashtable @{LogName='System'; Level=1,2; StartTime=$t.AddMinutes(-12); EndTime=$t.AddMinutes(1)} -ErrorAction SilentlyContinue | ForEach-Object {
  $m = ($_.Message -replace '\s+', ' ')
  if ($m.Length -gt 160) { $m = $m.Substring(0,160) }
  '{0} [{1}] {2}: {3}' -f $_.TimeCreated, $_.LevelDisplayName, $_.ProviderName, $m
}

Write-Output ''
Write-Output '=== Display driver (nvlddmkm) events, last 5 ==='
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'} -MaxEvents 5 -ErrorAction SilentlyContinue | ForEach-Object {
  '{0} [{1}] {2}' -f $_.TimeCreated, $_.LevelDisplayName, ($_.Message -replace '\s+', ' ')
}

Write-Output ''
Write-Output '=== WHEA hardware errors, last 5 ==='
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 5 -ErrorAction SilentlyContinue | ForEach-Object {
  $m = ($_.Message -replace '\s+', ' ')
  if ($m.Length -gt 200) { $m = $m.Substring(0,200) }
  '{0} [{1}] {2}' -f $_.TimeCreated, $_.LevelDisplayName, $m
}
