# Descarga el instalador de Python para Windows (64 bits) a la ruta indicada.
# Busca en python.org la version 3.13 mas reciente que tenga instalador y,
# si no puede, prueba con una lista de versiones conocidas.
param([string]$Destino)

$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$base = 'https://www.python.org/ftp/python/'

$candidatas = @()
try {
    $html = (Invoke-WebRequest -UseBasicParsing -Uri $base -TimeoutSec 30).Content
    $candidatas = @([regex]::Matches($html, '3\.13\.\d+(?=/)') |
        ForEach-Object { $_.Value } |
        Sort-Object { [version]$_ } -Descending -Unique)
} catch { }

$respaldo = @('3.13.9', '3.13.7', '3.13.5', '3.13.3', '3.12.10')
$versiones = @($candidatas) + @($respaldo) | Select-Object -Unique

foreach ($v in $versiones) {
    $url = "$base$v/python-$v-amd64.exe"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $Destino -TimeoutSec 600
        if ((Test-Path -LiteralPath $Destino) -and ((Get-Item -LiteralPath $Destino).Length -gt 1MB)) {
            Write-Host "      Descargado Python $v"
            exit 0
        }
    } catch { }
    if (Test-Path -LiteralPath $Destino) { Remove-Item -LiteralPath $Destino -Force }
}
exit 1
