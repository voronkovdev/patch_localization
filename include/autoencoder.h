#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Простой класс для извлечения дескрипторов
class Autoencoder {
public:
    Autoencoder() = default;

    // Извлекает 256-мерный дескриптор из патча
    // Использует простую HSV-гистограмму (8x8x4 = 256 измерений)
    std::vector<float> encode(const cv::Mat& patchBgr) const;

    // Создает аугментированные версии патча согласно ТЗ
    // 8 версий: оригинал + 4 поворота + 3 HSV варианта
    std::vector<cv::Mat> augmentPatch(const cv::Mat& patch) const;

    // Заглушки для будущих расширений
    bool train(const std::vector<cv::Mat>& trainPatches, int epochs, float learningRate, const std::string& modelPath);
    bool loadModel(const std::string& modelPath);
    std::vector<float> encodeWithModel(const cv::Mat& patchBgr) const;

private:
    std::string m_modelPath; // путь к модели (пока не используется)
};


