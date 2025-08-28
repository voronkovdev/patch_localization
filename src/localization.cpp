#include "patch_localization.h"
#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>

// L2 расстояние между векторами
double Localization::l2(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::cerr << "[ERROR] Размеры векторов не совпадают: " << a.size() << " vs " << b.size() << std::endl;
        return std::numeric_limits<double>::infinity();
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Поиск ближайшего дескриптора
int Localization::findNearest(const std::vector<float>& query, const std::vector<IndexedDescriptor>& db) {
    if (db.empty()) {
        return -1;
    }
    
    int bestIdx = 0;
    double bestDist = l2(query, db[0].descriptor);
    
    for (size_t i = 1; i < db.size(); ++i) {
        double dist = l2(query, db[i].descriptor);
        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = static_cast<int>(i);
        }
    }
    
    return bestIdx;
}

// Уточнение позиции с помощью корреляции
cv::Point Localization::refinePosition(const cv::Mat& bigImage, const cv::Mat& queryPatch, 
                                      const cv::Point& initialGuess, int searchRadius) {
    cv::Point bestPos = initialGuess;
    double bestScore = -1.0;
    
    // Многоуровневый поиск
    std::vector<std::pair<int, float>> searchLevels = {
        {searchRadius, 0.5f},    // Грубый поиск на уменьшенном изображении
        {searchRadius/2, 1.0f},  // Средний поиск на оригинальном размере
        {5, 1.0f}               // Точный поиск на оригинальном размере
    };
    
    for (const auto& level : searchLevels) {
        int radius = level.first;
        float scale = level.second;
        
        // Масштабируем изображения
        cv::Mat scaledBig, scaledPatch;
        cv::resize(bigImage, scaledBig, cv::Size(), scale, scale, cv::INTER_LINEAR);
        cv::resize(queryPatch, scaledPatch, cv::Size(), scale, scale, cv::INTER_LINEAR);
        
        // Масштабируем начальную позицию
        cv::Point scaledGuess(initialGuess.x * scale, initialGuess.y * scale);
        
        // Определяем область поиска
        int startX = std::max(0, scaledGuess.x - radius);
        int endX = std::min(scaledBig.cols - scaledPatch.cols, scaledGuess.x + radius);
        int startY = std::max(0, scaledGuess.y - radius);
        int endY = std::min(scaledBig.rows - scaledPatch.rows, scaledGuess.y + radius);
        
        // Ищем лучшее совпадение в области
        for (int y = startY; y <= endY; ++y) {
            for (int x = startX; x <= endX; ++x) {
                cv::Rect roi(x, y, scaledPatch.cols, scaledPatch.rows);
                cv::Mat region = scaledBig(roi);
                
                // Вычисляем корреляцию
                cv::Mat correlation1, correlation2, correlation3;
                cv::matchTemplate(region, scaledPatch, correlation1, cv::TM_CCOEFF_NORMED);
                cv::matchTemplate(region, scaledPatch, correlation2, cv::TM_CCORR_NORMED);
                cv::matchTemplate(region, scaledPatch, correlation3, cv::TM_SQDIFF_NORMED);
                
                double score1 = correlation1.at<float>(0, 0);
                double score2 = correlation2.at<float>(0, 0);
                double score3 = 1.0 - correlation3.at<float>(0, 0); // Инвертируем SQDIFF
                
                double combinedScore = (score1 + score2 + score3) / 3.0;
                
                if (combinedScore > bestScore) {
                    bestScore = combinedScore;
                    bestPos = cv::Point(x / scale, y / scale);
                }
            }
        }
        
        // Обновляем начальную позицию для следующего уровня
        // initialGuess = bestPos; // Убираем эту строку, так как initialGuess - const
    }
    
    return bestPos;
}

// Сохранение индекса
bool saveIndex(const std::string& path, const std::vector<IndexedDescriptor>& db) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Не удалось открыть файл для сохранения: " << path << std::endl;
        return false;
    }
    
    // Записываем количество записей
    size_t size = db.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    
    // Записываем каждую запись
    for (const auto& desc : db) {
        // Записываем позицию
        file.write(reinterpret_cast<const char*>(&desc.topLeft.x), sizeof(int));
        file.write(reinterpret_cast<const char*>(&desc.topLeft.y), sizeof(int));
        
        // Записываем тип аугментации
        file.write(reinterpret_cast<const char*>(&desc.augmentationType), sizeof(int));
        
        // Записываем размер дескриптора
        size_t descSize = desc.descriptor.size();
        file.write(reinterpret_cast<const char*>(&descSize), sizeof(size_t));
        
        // Записываем дескриптор
        file.write(reinterpret_cast<const char*>(desc.descriptor.data()), descSize * sizeof(float));
    }
    
    file.close();
    std::cout << "[INFO] Сохранен индекс: " << path << " (" << db.size() << " записей)" << std::endl;
    return true;
}

// Загрузка индекса
bool loadIndex(const std::string& path, std::vector<IndexedDescriptor>& db, int expectedDim) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Не удалось открыть файл для загрузки: " << path << std::endl;
        return false;
    }
    
    // Читаем количество записей
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    
    db.clear();
    db.reserve(size);
    
    // Читаем каждую запись
    for (size_t i = 0; i < size; ++i) {
        IndexedDescriptor desc;
        
        // Читаем позицию
        file.read(reinterpret_cast<char*>(&desc.topLeft.x), sizeof(int));
        file.read(reinterpret_cast<char*>(&desc.topLeft.y), sizeof(int));
        
        // Читаем тип аугментации
        file.read(reinterpret_cast<char*>(&desc.augmentationType), sizeof(int));
        
        // Читаем размер дескриптора
        size_t descSize;
        file.read(reinterpret_cast<char*>(&descSize), sizeof(size_t));
        
        // Проверяем размер
        if (descSize != static_cast<size_t>(expectedDim)) {
            std::cerr << "[ERROR] Неожиданный размер дескриптора: " << descSize << " (ожидалось " << expectedDim << ")" << std::endl;
            return false;
        }
        
        // Читаем дескриптор
        desc.descriptor.resize(descSize);
        file.read(reinterpret_cast<char*>(desc.descriptor.data()), descSize * sizeof(float));
        
        db.push_back(desc);
    }
    
    file.close();
    std::cout << "[INFO] Загружен индекс: " << path << " (" << db.size() << " записей)" << std::endl;
    return true;
}
