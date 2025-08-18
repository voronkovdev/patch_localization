#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Структура для хранения патча
struct Patch {
    cv::Mat image;            // 200x200 BGR изображение
    cv::Point topLeft;        // верхний левый угол в большом изображении
};

// Класс для обработки изображений
class ImageProcessor {
public:
    ImageProcessor() = default;

    // Загружает большое изображение
    bool loadImage(const std::string& path);
    
    // Возвращает загруженное изображение
    const cv::Mat& image() const;

    // Вырезает патчи из изображения
    std::vector<Patch> cutPatches(int patchSize, int step) const;

    // Простые аугментации
    static cv::Mat rotate90(const cv::Mat& src, int kTimes);
    static cv::Mat adjustHSV(const cv::Mat& src, double hueShiftDeg, double satScale, double valScale);

    // Рисует прямоугольник патча на изображении
    static void drawPatchRect(cv::Mat& canvas, const Patch& patch, const cv::Scalar& color, int thickness);

private:
    cv::Mat m_image;
};


