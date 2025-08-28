#include "patch_localization.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <tuple>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

// Генерирует демо-изображения для тестирования
static void generateDemoImages(const std::filesystem::path& outDir,
                               std::string& bigPath,
                               std::string& patchPath) {
    const int bigW = 10000; // согласно ТЗ: 10 км × 10 км
    const int bigH = 10000;
    const int patchSize = 200; // согласно ТЗ: 200м × 200м

    // Создаем большое изображение с градиентом и случайными фигурами
    cv::Mat big(bigH, bigW, CV_8UC3);
    
    // Градиентный фон
    for (int y = 0; y < bigH; ++y) {
        cv::Vec3b* row = big.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bigW; ++x) {
            row[x] = cv::Vec3b((x * 255) / bigW, (y * 255) / bigH, ((x + y) * 255) / (bigW + bigH));
        }
    }
    
    // Случайные линии и круги
    cv::RNG rng(12345);
    for (int i = 0; i < 200; ++i) {
        cv::Point p1(rng.uniform(0, bigW), rng.uniform(0, bigH));
        cv::Point p2(rng.uniform(0, bigW), rng.uniform(0, bigH));
        cv::Scalar color(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
        cv::line(big, p1, p2, color, 1);
    }
    
    for (int i = 0; i < 80; ++i) {
        cv::Point c(rng.uniform(100, bigW-100), rng.uniform(100, bigH-100));
        int radius = rng.uniform(10, 60);
        cv::Scalar color(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
        cv::circle(big, c, radius, color, 2);
    }

    // Выбираем позицию патча на сетке с шагом 50
    int x = 50 * ((rng.uniform(0, bigW - patchSize)) / 50);
    int y = 50 * ((rng.uniform(0, bigH - patchSize)) / 50);
    cv::Rect roi(x, y, patchSize, patchSize);
    cv::Mat patch = big(roi).clone();

    // Иногда поворачиваем патч для усложнения задачи
    int k = rng.uniform(0, 4);
    cv::Mat patchRot;
    if (k == 0) {
        patchRot = patch; // оставляем как есть
    } else if (k == 1) {
        cv::rotate(patch, patchRot, cv::ROTATE_90_CLOCKWISE);
    } else if (k == 2) {
        cv::rotate(patch, patchRot, cv::ROTATE_180);
    } else {
        cv::rotate(patch, patchRot, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    // Сохраняем изображения
    std::filesystem::create_directories(outDir);
    bigPath = (outDir / "demo_big.png").string();
    patchPath = (outDir / "demo_patch.png").string();
    cv::imwrite(bigPath, big);
    cv::imwrite(patchPath, patchRot);

    std::cout << "[DEMO] Большое изображение сохранено: " << bigPath << "\n";
    std::cout << "[DEMO] Патч сохранен: " << patchPath << "\n";
}

int main(int argc, char** argv) {
    // Устанавливаем кодировку для корректного отображения русских символов в Windows
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif
    
    std::string bigPath;
    std::string patchPath;
    std::string outDir = "output";
    bool demoMode = false;
    std::string evalCsv;
    int genPatchesN = 0;
    
    // Аргументы для сверточного автоэнкодера
    bool trainMode = false;
    bool rebuildIndex = false;
    std::string modelPath;
    int epochs = 100;
    float learningRate = 0.001f;

    // Парсим аргументы командной строки
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::cout << "Использование: patch_localization [опции]\n";
            std::cout << "Опции:\n";
            std::cout << "  --big <файл>           Путь к большому изображению\n";
            std::cout << "  --patch <файл>         Путь к патчу для поиска\n";
            std::cout << "  --out <папка>          Выходная папка (по умолчанию: output)\n";
            std::cout << "  --demo                 Демо-режим (генерирует тестовые изображения)\n";
            std::cout << "  --eval-csv <файл>      CSV файл для оценки\n";
            std::cout << "  --gen-patches <N>      Генерирует N патчей\n";
            std::cout << "  --train                Режим обучения сверточного автоэнкодера\n";
            std::cout << "  --rebuild-index        Принудительно пересоздать индекс\n";
            std::cout << "  --model <файл>         Путь к модели для загрузки/сохранения\n";
            std::cout << "  --epochs <число>       Количество эпох обучения (по умолчанию: 100)\n";
            std::cout << "  --lr <число>           Скорость обучения (по умолчанию: 0.001)\n";
            std::cout << "  --help, -h             Показать эту справку\n";
            return 0;
        }
        else if (a == "--big" && i + 1 < argc) bigPath = argv[++i];
        else if (a == "--patch" && i + 1 < argc) patchPath = argv[++i];
        else if (a == "--out" && i + 1 < argc) outDir = argv[++i];
        else if (a == "--demo") demoMode = true;
        else if (a == "--eval-csv" && i + 1 < argc) evalCsv = argv[++i];
        else if (a == "--gen-patches" && i + 1 < argc) { 
            try { genPatchesN = std::stoi(argv[++i]); } 
            catch (...) { genPatchesN = 0; } 
        }
        else if (a == "--train") trainMode = true;
        else if (a == "--rebuild-index") rebuildIndex = true;
        else if (a == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (a == "--epochs" && i + 1 < argc) { 
            try { epochs = std::stoi(argv[++i]); } 
            catch (...) { epochs = 100; } 
        }
        else if (a == "--lr" && i + 1 < argc) { 
            try { learningRate = std::stof(argv[++i]); } 
            catch (...) { learningRate = 0.001f; } 
        }
    }

    // Создаем выходную папку и лог-файл
    fs::create_directories(outDir);
    std::ofstream runLog((fs::path(outDir) / "run.log").string(), std::ios::out | std::ios::trunc);
    auto logmsg = [&](const std::string& m){ 
        std::cout << m << "\n"; 
        if (runLog.good()) { 
            runLog << m << "\n"; 
            runLog.flush(); 
        } 
    };
    
    logmsg("[INFO] Система локализации патчей запускается...");

    // Демо-режим
    if (demoMode) {
        std::string demoBig, demoPatch;
        logmsg("[INFO] Демо-режим: генерируем изображения...");
        generateDemoImages(outDir, demoBig, demoPatch);
        bigPath = demoBig;
        // Устанавливаем patchPath только если не указан eval-csv
        if (evalCsv.empty()) {
            patchPath = demoPatch;
        }
    }

    // Проверяем аргументы
    if (bigPath.empty() || (patchPath.empty() && evalCsv.empty() && genPatchesN == 0 && !trainMode)) {
        logmsg("[ERROR] Неверные аргументы.");
        std::cerr << "Использование: patch_localization --big <большое_изображение> --patch <патч> [--out выходная_папка]\n";
        std::cerr << "   или: patch_localization --demo [--out выходная_папка]\n";
        std::cerr << "   или: patch_localization --big <большое_изображение> --eval-csv <csv_файл> [--out выходная_папка]\n";
        std::cerr << "   или: patch_localization --big <большое_изображение> --gen-patches <N> [--out выходная_папка]\n";
        std::cerr << "   или: patch_localization --train --big <большое_изображение> [--epochs N] [--lr X] [--model файл] [--out выходная_папка]\n";
        std::cerr << "   или: patch_localization --help для получения справки\n";
        return 1;
    }
    
    logmsg(std::string("[INFO] Путь к большому изображению: ") + bigPath);
    if (!patchPath.empty()) logmsg(std::string("[INFO] Путь к патчу: ") + patchPath);
    if (!evalCsv.empty()) logmsg(std::string("[INFO] CSV для оценки: ") + evalCsv);

    // Загружаем большое изображение
    ImageProcessor ip;
    if (!ip.loadImage(bigPath)) {
        logmsg(std::string("[ERROR] Не удалось загрузить большое изображение: ") + bigPath);
        return 2;
    }

    // Вырезаем патчи согласно ТЗ: 200×200 пикселей с шагом 50
    const int patchSize = 200; // согласно ТЗ: 200м × 200м
    const int step = 50;
    auto patches = ip.cutPatches(patchSize, step);
    logmsg("[INFO] Вырезано патчей: " + std::to_string(patches.size()));

    // Создаем сверточный автоэнкодер
    ConvolutionalAutoencoder convAE;
    
    // Загружаем обученную модель (если указана)
    if (!modelPath.empty()) {
        if (convAE.loadModel(modelPath)) {
            logmsg("[INFO] Загружена обученная модель: " + modelPath);
        } else {
            logmsg("[WARN] Не удалось загрузить модель: " + modelPath + ", будет создана новая");
        }
    }
    
    std::vector<IndexedDescriptor> db;
    
    // Индекс зависит от имени файла большого изображения и времени его изменения
    std::string indexPath;
    try {
        auto ft = fs::last_write_time(bigPath);
        uint64_t mtimeTicks = static_cast<uint64_t>(ft.time_since_epoch().count());
        std::string baseName = fs::path(bigPath).filename().string();
        indexPath = (fs::path(outDir) / (std::string("index_") + baseName + "_" + std::to_string(mtimeTicks) + ".pldb")).string();
    } catch (...) {
        indexPath = (fs::path(outDir) / "index.pldb").string();
    }
    
    if (!rebuildIndex && loadIndex(indexPath, db, 256)) {
        logmsg("[INFO] Загружен индекс: " + indexPath + ", записей=" + std::to_string(db.size()));
    } else {
        // Создаем новый индекс с аугментированными версиями
        logmsg("[INFO] Создаем аугментированную базу данных для сверточного автоэнкодера...");
        
        // Каждый патч создает 8 версий: оригинал + 3 поворота + 4 HSV варианта
        size_t totalPatches = patches.size() * 8;
        db.reserve(totalPatches);
        
        // Показываем прогресс каждые 1000 патчей
        size_t progressStep = 1000;
        size_t processed = 0;
        
        logmsg("[DEBUG] Начинаем обработку " + std::to_string(patches.size()) + " патчей...");
        
        for (const auto& p : patches) {
            // Получаем аугментированные версии патча
            auto augmented = convAE.augmentPatch(p.image);
            
            for (size_t augIdx = 0; augIdx < augmented.size(); ++augIdx) {
                IndexedDescriptor id;
                id.topLeft = p.topLeft; // позиция остается та же
                id.augmentationType = static_cast<int>(augIdx); // тип аугментации
                
                // Кодируем патч с помощью сверточного автоэнкодера
                id.descriptor = convAE.encode(augmented[augIdx]);
                db.push_back(std::move(id));
                
                processed++;
                if (processed % progressStep == 0) {
                    logmsg("[INFO] Обработано патчей: " + std::to_string(processed) + "/" + std::to_string(totalPatches));
                }
            }
        }
        
        logmsg("[INFO] Создано " + std::to_string(db.size()) + " аугментированных дескрипторов");
        
        if (saveIndex(indexPath, db)) logmsg("[INFO] Сохранен индекс: " + indexPath);
        else logmsg("[WARN] Не удалось сохранить индекс: " + indexPath);
    }

    // Обучение сверточного автоэнкодера (если включено)
    if (trainMode) {
        if (modelPath.empty()) {
            modelPath = (fs::path(outDir) / "convolutional_autoencoder.model").string();
        }
        
        logmsg("[INFO] Начинаем обучение сверточного автоэнкодера...");
        logmsg("[INFO] Эпохи: " + std::to_string(epochs));
        logmsg("[INFO] Скорость обучения: " + std::to_string(learningRate));
        logmsg("[INFO] Путь к модели: " + modelPath);
        
        // Подготавливаем данные для обучения (используем все патчи)
        std::vector<cv::Mat> trainPatches;
        trainPatches.reserve(patches.size());
        for (size_t i = 0; i < patches.size(); ++i) {
            trainPatches.push_back(patches[i].image);
        }
        
        logmsg("[INFO] Патчей для обучения: " + std::to_string(trainPatches.size()));
        
        // Обучаем сеть
        if (convAE.train(trainPatches, epochs, learningRate)) {
            logmsg("[INFO] Обучение завершено успешно!");
            if (convAE.saveModel(modelPath)) {
                logmsg("[INFO] Модель сохранена: " + modelPath);
            }
        } else {
            logmsg("[WARN] Обучение не удалось");
        }
    }

    // Функция для поиска лучшего совпадения
    auto solveOne = [&](const cv::Mat& queryImg) {
        int bestIdx = -1; 
        double bestDist = std::numeric_limits<double>::infinity(); 
        
        // Получаем дескриптор запроса с помощью сверточного автоэнкодера
        auto qdesc = convAE.encode(queryImg);
        
        // Ищем ближайший в аугментированной базе данных
        int idx = Localization::findNearest(qdesc, db);
        if (idx >= 0) {
            bestIdx = idx;
            bestDist = Localization::l2(qdesc, db[idx].descriptor);
        }
        
        return std::tuple<int,double,int>(bestIdx, bestDist, 0);
    };

    // Обрабатываем патч
    if (!patchPath.empty()) {
        cv::Mat query = cv::imread(patchPath, cv::IMREAD_COLOR);
        if (query.empty()) {
            logmsg(std::string("[ERROR] Не удалось загрузить патч: ") + patchPath);
            return 3;
        }
        
        // Приводим к нужному размеру
        if (query.cols != patchSize || query.rows != patchSize) {
            cv::resize(query, query, cv::Size(patchSize, patchSize), 0, 0, cv::INTER_AREA);
        }
        
        // Ищем лучшее совпадение
        auto result = solveOne(query);
        int bestIdx = std::get<0>(result);
        double bestDist = std::get<1>(result);
        int bestRot = std::get<2>(result);
        if (bestIdx < 0) { 
            logmsg("[ERROR] Совпадение не найдено (пустая база данных)."); 
            return 4; 
        }
        
        // Визуализируем результат
        cv::Mat vis = ip.image().clone();
        
        // Вычисляем индекс оригинального патча (каждый патч имеет 8 аугментированных версий)
        int originalPatchIdx = bestIdx / 8;
        
        if (originalPatchIdx >= patches.size()) {
            logmsg("[ERROR] originalPatchIdx (" + std::to_string(originalPatchIdx) + ") >= patches.size (" + std::to_string(patches.size()) + ")");
            return 4;
        }
        
        Patch bestPatch{ patches[originalPatchIdx].image, patches[originalPatchIdx].topLeft };
        ImageProcessor::drawPatchRect(vis, bestPatch, cv::Scalar(0, 0, 255), 2);
        cv::imwrite((fs::path(outDir) / "localization_result.png").string(), vis);
        
        logmsg("[INFO] Лучшее совпадение верхний-левый: (" + std::to_string(bestPatch.topLeft.x) + ", " + std::to_string(bestPatch.topLeft.y) + ")");
        
        // Показываем информацию об аугментации
        if (bestIdx < db.size()) {
            std::string augInfo;
            switch (db[bestIdx].augmentationType) {
                case 0: augInfo = "оригинал"; break;
                case 1: augInfo = "поворот 90°"; break;
                case 2: augInfo = "поворот 180°"; break;
                case 3: augInfo = "поворот 270°"; break;
                case 4: augInfo = "HSV вариант 1"; break;
                case 5: augInfo = "HSV вариант 2"; break;
                case 6: augInfo = "HSV вариант 3"; break;
                case 7: augInfo = "HSV вариант 4"; break;
                default: augInfo = "неизвестно"; break;
            }
            logmsg("[INFO] Тип аугментации найденного патча: " + augInfo);
        }
        
        logmsg(std::string("[INFO] Визуализация сохранена в: ") + (fs::path(outDir) / "localization_result.png").string());
        return 0;
    }

    // Генерация патчей для оценки
    if (genPatchesN > 0) {
        fs::create_directories(fs::path(outDir) / "patches");
        std::ofstream csv((fs::path(outDir) / "eval.csv").string());
        csv << "patch_path,x,y\n";
        
        cv::RNG rng(123456);
        int width = ip.image().cols, height = ip.image().rows;
        
        for (int i = 0; i < genPatchesN; ++i) {
            // Выбираем позицию на сетке с шагом 50
            int maxX = (1 > width - patchSize) ? 1 : width - patchSize;
            int maxY = (1 > height - patchSize) ? 1 : height - patchSize;
            int x = 50 * (rng.uniform(0, maxX) / 50);
            int y = 50 * (rng.uniform(0, maxY) / 50);
            cv::Rect roi(x, y, patchSize, patchSize);
            cv::Mat p = ip.image()(roi).clone();
            
            std::string pth = (fs::path(outDir) / "patches" / ("p_" + std::to_string(i) + ".png")).string();
            cv::imwrite(pth, p);
            
            int cx = x + patchSize/2; 
            int cy = y + patchSize/2;
            csv << pth << "," << cx << "," << cy << "\n";
        }
        csv.close();
        logmsg("[INFO] Сгенерированы патчи и CSV в: " + (fs::path(outDir) / "eval.csv").string());
        return 0;
    }

    // Оценка по CSV
    if (!evalCsv.empty()) {
        std::ifstream fin(evalCsv);
        if (!fin.good()) { 
            logmsg("[ERROR] Не удается открыть CSV: " + evalCsv); 
            return 5; 
        }
        
        std::string line; 
        bool headerSkipped = false; 
        size_t total = 0; 
        double mae = 0.0; 
        size_t ok20 = 0;
        
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            if (!headerSkipped) { 
                headerSkipped = true; 
                if (line.find("patch_path") != std::string::npos) continue; 
            }
            
            std::stringstream ss(line);
            std::string patchFile, xs, ys;
            if (!std::getline(ss, patchFile, ',')) continue;
            if (!std::getline(ss, xs, ',')) continue;
            if (!std::getline(ss, ys, ',')) continue;
            
            cv::Mat q = cv::imread(patchFile, cv::IMREAD_COLOR);
            if (q.empty()) { 
                logmsg("[WARN] Пропускаем (не удается прочитать): " + patchFile); 
                continue; 
            }
            
            if (q.cols != patchSize || q.rows != patchSize) {
                cv::resize(q, q, cv::Size(patchSize, patchSize), 0, 0, cv::INTER_AREA);
            }
            
            auto result = solveOne(q);
            int bestIdx = std::get<0>(result);
            double bestDist = std::get<1>(result);
            int bestRot = std::get<2>(result);
            if (bestIdx < 0) { 
                logmsg("[WARN] Совпадение не найдено для: " + patchFile); 
                continue; 
            }
            
            int originalPatchIdx = bestIdx / 8;
            if (originalPatchIdx < 0 || originalPatchIdx >= static_cast<int>(patches.size())) {
                logmsg("[WARN] Индекс вне диапазона при вычислении центра предсказания");
                continue;
            }
            
            // Получаем начальную позицию
            cv::Point initialPos = patches[originalPatchIdx].topLeft;
            
            // Применяем постобработку для уточнения позиции
            cv::Point refinedPos = Localization::refinePosition(ip.image(), q, initialPos, 25);
            
            cv::Point predCenter(refinedPos.x + patchSize/2, refinedPos.y + patchSize/2);
            double gx = 0.0, gy = 0.0; 
            try { 
                gx = std::stod(xs); 
                gy = std::stod(ys); 
            } catch (...) { 
                continue; 
            }
            
            double err = std::hypot(predCenter.x - gx, predCenter.y - gy);
            mae += err; 
            total += 1; 
            if (err <= 20.0) ok20 += 1;
        }
        
        if (total == 0) { 
            logmsg("[ERROR] Нет валидных строк в CSV."); 
            return 6; 
        }
        
        double meanAbsErr = mae / static_cast<double>(total);
        double frac20 = static_cast<double>(ok20) / static_cast<double>(total);
        
        std::ofstream mf((fs::path(outDir) / "metrics.txt").string());
        mf << "samples=" << total << "\n";
        mf << "mae_pixels=" << meanAbsErr << "\n";
        mf << "frac_err_le_20=" << frac20 << "\n";
        mf.close();
        
        logmsg("[INFO] Оценка завершена. Образцов=" + std::to_string(total) + 
               ", MAE=" + std::to_string(meanAbsErr) + 
               ", доля<=20=" + std::to_string(frac20));
        return 0;
    }

    return 0;
}
