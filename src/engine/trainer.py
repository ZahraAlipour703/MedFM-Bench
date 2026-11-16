Set-Location "D:\zra\PROJECTS\Github\MedFM-Bench"

if ((Get-Location).Path -ne "D:\zra\PROJECTS\Github\MedFM-Bench") {
    throw "Wrong repository."
}

New-Item -ItemType Directory -Force .\src\engine | Out-Null
New-Item -ItemType File -Force .\src\engine\__init__.py | Out-Null