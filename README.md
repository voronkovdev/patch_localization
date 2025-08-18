## Локализация патча (C++/OpenCV + TinyDNN)

Простой и готовый к использованию базовый проект локализации патчей, соответствующий ТЗ:
- Размер патча 200×200, шаг 50 пикселей
- Аугментации: 8 на патч (оригинал, 3 поворота, 4 варианта HSV)
- Дескриптор 256 измерений: латентный вектор сверточного автоэнкодера TinyDNN или запасной вариант — HSV-гистограмма
- Поиск ближайшего соседа в пространстве дескрипторов (L2)
- Метрики: MAE (в пикселях) и доля запросов с ошибкой ≤ 20 пикселей

### 1) Требования
- CMake 3.15+
- MSVC (Visual Studio 2019/2022)
- OpenCV 4.x (при необходимости поправьте путь)
- TinyDNN (включен в `third_party/tiny-dnn-master`)

### 2) Сборка (Windows)
Откройте PowerShell в корне проекта и выполните:
```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
Заметки:
- Путь к OpenCV задан в `CMakeLists.txt` через `OpenCV_DIR`. При необходимости укажите ваш путь.
- При включенном TinyDNN (по умолчанию) сборка использует `/bigobj` (уже настроено).

### 3) Запуск: быстрый старт
- Демо (синтетическое большое изображение + патч):
```powershell
./build/Release/patch_localization.exe --demo --out output_dir
```
- Поиск на своих данных:
```powershell
./build/Release/patch_localization.exe --big path\to\big_image.png --patch path\to\patch.png --out output_dir
```
- Генерация CSV для оценки (патчи берутся из `--big`) и сохранение изображений:
```powershell
./build/Release/patch_localization.exe --big path\to\big_image.png --gen-patches 200 --out output_dir
```
- Оценка (подсчет метрик по CSV):
```powershell
./build/Release/patch_localization.exe --big path\to\big_image.png --eval-csv output_dir\eval.csv --out output_dir
```

### 4) Обучение автоэнкодера (TinyDNN)
- Обучение на патчах, вырезанных из `--big` (первые 100 патчей для скорости):
```powershell
./build/Release/patch_localization.exe --train --big path\to\big_image.png --epochs 10 --lr 0.001 --model output_dir\autoencoder.model --out output_dir
```
- Использование обученной модели для извлечения дескрипторов:
```powershell
./build/Release/patch_localization.exe --big path\to\big_image.png --patch path\to\patch.png --model output_dir\autoencoder.model --out output_dir
```

Под капотом:
- Большое изображение режется на патчи 200×200 с шагом 50
- Для каждого патча создаются 8 аугментаций (оригинал, 3 поворота, 4 HSV)
- Дескрипторы считаются автоэнкодером TinyDNN (если модель загружена) или HSV-гистограммой (fallback)
- Дескрипторы кэшируются в `output_dir/index.pldb` (быстрая перезагрузка)
- Возвращается ближайший сосед по L2; визуализация — `output_dir/localization_result.png`

### 5) Справка по CLI
Запустите с `--help` для просмотра всех опций. Кратко:
- `--big <файл>`: путь к большому изображению
- `--patch <файл>`: путь к одному патчу для поиска
- `--out <папка>`: выходная папка
- `--demo`: сгенерировать синтетические большое изображение и патч
- `--gen-patches <N>`: сэмплировать N патчей и записать `eval.csv`
- `--eval-csv <файл>`: посчитать MAE и долю ≤ 20 пикселей
- `--train`: обучить автоэнкодер TinyDNN на вырезанных патчах
- `--model <файл>`: путь к модели для загрузки/сохранения
- `--epochs <N>`: количество эпох (по умолчанию 100)
- `--lr <X>`: скорость обучения (по умолчанию 0.001)

### 6) Датасет University1652 (опционально)
Если датасет находится по пути `C:\Users\voron\Desktop\University1652-Baseline-1.2`:
- В качестве большого изображения можно использовать любое 4K-изображение, например `image_4K\00\query.png` или `image_4K\*.png`
- Далее запустите `--gen-patches` и `--eval-csv`, как показано выше, чтобы проверить метрики на случайно выбранных патчах из этого изображения

Пример:
```powershell
./build/Release/patch_localization.exe --big C:\Users\voron\Desktop\University1652-Baseline-1.2\image_4K\00\query.png --gen-patches 300 --out output_dir
./build/Release/patch_localization.exe --big C:\Users\voron\Desktop\University1652-Baseline-1.2\image_4K\00\query.png --eval-csv output_dir\eval.csv --out output_dir
```

### 7) Технические заметки (соответствие ТЗ)
- Размер патча 200×200, шаг 50
- Аугментации: 8 на исходный патч (оригинал, 3 поворота на 90°, 4 варианта HSV)
- Дескриптор 256 измерений:
  - Сверточный автоэнкодер TinyDNN, извлечение латентного вектора во время прямого прохода (после FC(256))
  - Запасной вариант — HSV-гистограмма (8×8×4)
- L2-метрика для поиска ближайшего соседа
- Метрики: MAE (пиксели) и доля образцов с ошибкой ≤ 20 пикселей

### 8) Частые проблемы
- Не найден OpenCV: обновите `OpenCV_DIR` в `CMakeLists.txt`
- Краказябы в консоли Windows: приложение включает UTF-8 (`SetConsoleOutputCP(CP_UTF8)`) автоматически
- Медленный первый запуск: индекс дескрипторов кэшируется в `output_dir/index.pldb`

- Постоянная настройка UTF-8 в PowerShell (чтобы не вводить команды каждый раз):
  - Откройте PowerShell от имени пользователя и выполните:
  ```powershell
  # Разрешаем запуск профильных скриптов для текущего пользователя
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

  # Профиль для Windows PowerShell (5.x)
  $p1 = Join-Path $env:USERPROFILE 'Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1'
  New-Item -Type Directory -Force -Path (Split-Path -Parent $p1) | Out-Null
  Add-Content -Path $p1 -Value "# UTF-8 auto`nchcp 65001 > `$null`n[Console]::InputEncoding=[System.Text.UTF8Encoding]::new()`n[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new()`n`$OutputEncoding=[System.Text.UTF8Encoding]::new()`n"

  # Профиль для PowerShell 7+
  $p2 = Join-Path $env:USERPROFILE 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1'
  New-Item -Type Directory -Force -Path (Split-Path -Parent $p2) | Out-Null
  Add-Content -Path $p2 -Value "# UTF-8 auto`n[Console]::InputEncoding=[System.Text.UTF8Encoding]::new()`n[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new()`n`$OutputEncoding=[System.Text.UTF8Encoding]::new()`n"
  ```
  - Перезапустите PowerShell. Теперь вывод по умолчанию будет в UTF‑8.

### 9) Выходные файлы
- `output_dir/localization_result.png` — визуализация лучшего совпадения
- `output_dir/index.pldb` — кэш дескрипторов (с типом аугментации)
- `output_dir/eval.csv` — сгенерированные патчи + истинные центры
- `output_dir/metrics.txt` — MAE и доля ≤ 20 пикселей
- `output_dir/run.log` — логи запуска и диагностические сообщения

### 10) Лицензии
- TinyDNN и прочие сторонние компоненты имеют свои лицензии в каталоге `third_party/`

