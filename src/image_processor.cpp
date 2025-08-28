#include "patch_localization.h"
#include <iostream>

bool ImageProcessor::loadImage(const std::string& path) {
    image_ = cv::imread(path, cv::IMREAD_COLOR);
    if (image_.empty()) {
        std::cerr << "[ERROR] Не удалось загрузить изображение: " << path << std::endl;
        return false;
    }
    std::cout << "[INFO] Загружено изображение: " << path << " (" << image_.cols << "x" << image_.rows << ")" << std::endl;
    return true;
}

std::vector<Patch> ImageProcessor::cutPatches(int patchSize, int step) {
    std::vector<Patch> patches;
    
    int width = image_.cols;
    int height = image_.rows;
    
    // Вырезаем патчи с шагом step
    for (int y = 0; y <= height - patchSize; y += step) {
        for (int x = 0; x <= width - patchSize; x += step) {
            cv::Rect roi(x, y, patchSize, patchSize);
            cv::Mat patch = image_(roi).clone();
            
            Patch p;
            p.image = patch;
            p.topLeft = cv::Point(x, y);
            patches.push_back(p);
        }
    }
    
    std::cout << "[INFO] Вырезано патчей: " << patches.size() << std::endl;
    return patches;
}

void ImageProcessor::drawPatchRect(cv::Mat& img, const Patch& patch, const cv::Scalar& color, int thickness) {
    cv::Rect rect(patch.topLeft.x, patch.topLeft.y, patch.image.cols, patch.image.rows);
    cv::rectangle(img, rect, color, thickness);
}
