#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Структура для хранения индексированного дескриптора
struct IndexedDescriptor {
    cv::Point topLeft;        // позиция патча в большом изображении
    std::vector<float> descriptor;  // 256-мерный дескриптор
    int augmentationType;     // тип аугментации: 0=оригинал, 1-3=повороты, 4-6=HSV
};

// Класс для локализации патчей
class Localization {
public:
    // Вычисляет L2 расстояние между двумя дескрипторами
    static double l2(const std::vector<float>& a, const std::vector<float>& b);

    // Находит ближайший патч в базе данных
    static int findNearest(const std::vector<float>& query,
                           const std::vector<IndexedDescriptor>& database);
};


