#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Структура для хранения патча с позицией
struct Patch {
    cv::Mat image;
    cv::Point topLeft;
};

// Структура для индексированного дескриптора
struct IndexedDescriptor {
    cv::Point topLeft;
    std::vector<float> descriptor;
    int augmentationType; // 0=оригинал, 1-3=повороты, 4-7=HSV варианты
};

// Класс для обработки изображений
class ImageProcessor {
public:
    bool loadImage(const std::string& path);
    std::vector<Patch> cutPatches(int patchSize, int step);
    const cv::Mat& image() const { return image_; }
    static void drawPatchRect(cv::Mat& img, const Patch& patch, const cv::Scalar& color, int thickness);

private:
    cv::Mat image_;
};

// Сверточный автоэнкодер согласно ТЗ
class ConvolutionalAutoencoder {
public:
    ConvolutionalAutoencoder();
    
    // Обучение
    bool train(const std::vector<cv::Mat>& patches, int epochs, float learningRate);
    
    // Инференс
    std::vector<float> encode(const cv::Mat& patch);  // 256-мерное латентное представление
    cv::Mat decode(const std::vector<float>& latent); // Реконструкция патча
    
    // Аугментация данных согласно ТЗ
    std::vector<cv::Mat> augmentPatch(const cv::Mat& patch) const;
    
    // Сохранение/загрузка модели
    bool saveModel(const std::string& path);
    bool loadModel(const std::string& path);

private:
    // Энкодер: 32x32x3 -> 16x16x32 -> 8x8x64 -> 4x4x128 -> 256
    cv::Mat conv1_, conv2_, conv3_;  // Веса сверточных слоев
    cv::Mat latent_;                 // Веса латентного слоя
    
    // Декодер: 256 -> 4x4x128 -> 8x8x64 -> 16x16x32 -> 32x32x3
    cv::Mat latentDec_;              // Веса декодера латентного слоя
    cv::Mat deconv1_, deconv2_, deconv3_; // Веса деконволюции
    
    // Градиенты для backpropagation
    cv::Mat conv1_grad_, conv2_grad_, conv3_grad_;
    cv::Mat latent_grad_;
    cv::Mat latentDec_grad_;
    cv::Mat deconv1_grad_, deconv2_grad_, deconv3_grad_;
    
    // Промежуточные активации для backprop
    std::vector<cv::Mat> activations_;
    std::vector<cv::Mat> pre_activations_;
    
    // Вспомогательные функции
    cv::Mat forwardPass(const cv::Mat& input);
    void backwardPass(const cv::Mat& input, const cv::Mat& target);
    void updateWeights(float learningRate);
    float computeLoss(const cv::Mat& original, const cv::Mat& reconstructed);
    
    // Функции для backpropagation
    cv::Mat conv2d(const cv::Mat& input, const cv::Mat& weights, int stride = 1, int padding = 1);
    cv::Mat conv2dBackward(const cv::Mat& gradOutput, const cv::Mat& input, const cv::Mat& weights, 
                          cv::Mat& weightGrad, int stride = 1, int padding = 1);
    cv::Mat maxPool2d(const cv::Mat& input, int kernelSize = 2, int stride = 2);
    cv::Mat maxPool2dBackward(const cv::Mat& gradOutput, const cv::Mat& input, int kernelSize = 2, int stride = 2);
    std::vector<float> denseForward(const std::vector<float>& input, const cv::Mat& weights);
    std::vector<float> denseBackward(const std::vector<float>& gradOutput, const std::vector<float>& input, 
                                   const cv::Mat& weights, cv::Mat& weightGrad);
};

// Функции локализации
namespace Localization {
    double l2(const std::vector<float>& a, const std::vector<float>& b);
    int findNearest(const std::vector<float>& query, const std::vector<IndexedDescriptor>& db);
    cv::Point refinePosition(const cv::Mat& bigImage, const cv::Mat& queryPatch, 
                           const cv::Point& initialGuess, int searchRadius = 25);
}

// Функции для работы с индексом
bool saveIndex(const std::string& path, const std::vector<IndexedDescriptor>& db);
bool loadIndex(const std::string& path, std::vector<IndexedDescriptor>& db, int expectedDim);
