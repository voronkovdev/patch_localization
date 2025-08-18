#include "image_processor.h"

// Загружает изображение из файла
bool ImageProcessor::loadImage(const std::string& path) {
    m_image = cv::imread(path, cv::IMREAD_COLOR);
    return !m_image.empty();
}

// Возвращает загруженное изображение
const cv::Mat& ImageProcessor::image() const {
    return m_image;
}

// Вырезает патчи из изображения с заданным шагом
std::vector<Patch> ImageProcessor::cutPatches(int patchSize, int step) const {
    std::vector<Patch> patches;
    if (m_image.empty()) return patches;

    const int width = m_image.cols;
    const int height = m_image.rows;

    // Простой алгоритм: идем по сетке с шагом step
    for (int y = 0; y + patchSize <= height; y += step) {
        for (int x = 0; x + patchSize <= width; x += step) {
            cv::Rect roi(x, y, patchSize, patchSize);
            Patch p;
            p.image = m_image(roi).clone();
            p.topLeft = cv::Point(x, y);
            patches.push_back(std::move(p));
        }
    }
    return patches;
}

// Поворачивает изображение на 90 градусов k раз
cv::Mat ImageProcessor::rotate90(const cv::Mat& src, int kTimes) {
    CV_Assert(!src.empty());
    int k = ((kTimes % 4) + 4) % 4;  // нормализуем к [0,3]
    cv::Mat dst = src.clone();
    
    switch (k) {
        case 0: return dst;  // без поворота
        case 1: cv::rotate(src, dst, cv::ROTATE_90_CLOCKWISE); break;      // 90°
        case 2: cv::rotate(src, dst, cv::ROTATE_180); break;               // 180°
        case 3: cv::rotate(src, dst, cv::ROTATE_90_COUNTERCLOCKWISE); break; // 270°
    }
    return dst;
}

// Простая аугментация HSV
cv::Mat ImageProcessor::adjustHSV(const cv::Mat& src, double hueShiftDeg, double satScale, double valScale) {
    CV_Assert(!src.empty());
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    // Простое изменение: сдвигаем оттенок, масштабируем насыщенность и яркость
    const int shift = static_cast<int>(std::round(hueShiftDeg / 2.0)); // 1 шаг = 2 градуса
    
    for (int y = 0; y < hsv.rows; ++y) {
        cv::Vec3b* row = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv.cols; ++x) {
            // Оттенок (H) - сдвигаем
            int h = static_cast<int>(row[x][0]) + shift;
            h %= 180;  // оборачиваем к [0,179]
            if (h < 0) h += 180;
            row[x][0] = static_cast<uchar>(h);

            // Насыщенность (S) - масштабируем
            int s = static_cast<int>(std::round(row[x][1] * satScale));
            if (s < 0) s = 0; else if (s > 255) s = 255;
            row[x][1] = static_cast<uchar>(s);

            // Яркость (V) - масштабируем
            int v = static_cast<int>(std::round(row[x][2] * valScale));
            if (v < 0) v = 0; else if (v > 255) v = 255;
            row[x][2] = static_cast<uchar>(v);
        }
    }
    
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}

// Рисует прямоугольник патча на изображении
void ImageProcessor::drawPatchRect(cv::Mat& canvas, const Patch& patch, const cv::Scalar& color, int thickness) {
    cv::Rect roi(patch.topLeft.x, patch.topLeft.y, patch.image.cols, patch.image.rows);
    cv::rectangle(canvas, roi, color, thickness);
}


