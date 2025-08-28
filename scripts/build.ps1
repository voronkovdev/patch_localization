# Скрипт сборки проекта patch_localization
# Использует CMake для сборки

Write-Host "Сборка проекта patch_localization..." -ForegroundColor Green

# Создаем папку для сборки
if (!(Test-Path "build")) {
    New-Item -ItemType Directory -Path "build"
    Write-Host "Создана папка build" -ForegroundColor Yellow
}

# Переходим в папку сборки
Set-Location "build"

# Конфигурируем проект с CMake
Write-Host "Конфигурирование CMake..." -ForegroundColor Yellow
cmake .. -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка конфигурации CMake!" -ForegroundColor Red
    Set-Location ".."
    exit 1
}

# Собираем проект
Write-Host "Сборка проекта..." -ForegroundColor Yellow
cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка сборки!" -ForegroundColor Red
    Set-Location ".."
    exit 1
}

Write-Host "Сборка завершена успешно!" -ForegroundColor Green
Write-Host "Исполняемый файл: build\Release\patch_localization.exe" -ForegroundColor Cyan

# Возвращаемся в корневую папку
Set-Location ".."
