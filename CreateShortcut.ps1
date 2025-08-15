Write-Host "Création d'un raccourci pour l'application d'analyse vidéo..." -ForegroundColor Cyan

# Définition des chemins
$currentDir = $PSScriptRoot
if (!$currentDir) { $currentDir = Get-Location }

$shortcutName = "Analyse IA YouTube.lnk"
$desktopPath = [System.Environment]::GetFolderPath("Desktop")
$targetFile = Join-Path $currentDir "START.bat"
$iconLocation = "$env:SystemRoot\System32\SHELL32.dll,46"

# Création du raccourci
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$desktopPath\$shortcutName")
$Shortcut.TargetPath = "cmd.exe"
$Shortcut.Arguments = "/c `"$targetFile`""
$Shortcut.WorkingDirectory = $currentDir
$Shortcut.IconLocation = $iconLocation
$Shortcut.Description = "Système d'analyse vidéo YouTube avec IA"
$Shortcut.Save()

# Vérification de la création du raccourci
if (Test-Path "$desktopPath\$shortcutName") {
    Write-Host "✅ Raccourci créé avec succès sur votre bureau!" -ForegroundColor Green
    Write-Host "   Vous pouvez maintenant lancer l'application en double-cliquant sur '$shortcutName'" -ForegroundColor Yellow
} else {
    Write-Host "❌ Erreur lors de la création du raccourci." -ForegroundColor Red
}

Write-Host "`nAppuyez sur une touche pour quitter..." -ForegroundColor Magenta
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
