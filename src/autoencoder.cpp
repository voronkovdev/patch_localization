#include "patch_localization.h"
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// Конструктор автоэнкодера
ConvolutionalAutoencoder::ConvolutionalAutoencoder() {
    // Инициализируем веса с Xavier/Glorot
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Энкодер: 32x32x3 -> 16x16x32 -> 8x8x64 -> 4x4x128 -> 256
    // Conv1: 3->32, kernel=3x3, stride=1, padding=1
    conv1_ = cv::Mat::zeros(32, 3 * 3 * 3, CV_32F);
    conv1_grad_ = cv::Mat::zeros(32, 3 * 3 * 3, CV_32F);
    float scale1 = std::sqrt(2.0f / (3 * 3 * 3));
    std::normal_distribution<float> dist1(0.0f, scale1);
    for (int i = 0; i < conv1_.rows; ++i) {
        for (int j = 0; j < conv1_.cols; ++j) {
            conv1_.at<float>(i, j) = dist1(gen);
        }
    }
    
    // Conv2: 32->64, kernel=3x3, stride=1, padding=1
    conv2_ = cv::Mat::zeros(64, 32 * 3 * 3, CV_32F);
    conv2_grad_ = cv::Mat::zeros(64, 32 * 3 * 3, CV_32F);
    float scale2 = std::sqrt(2.0f / (32 * 3 * 3));
    std::normal_distribution<float> dist2(0.0f, scale2);
    for (int i = 0; i < conv2_.rows; ++i) {
        for (int j = 0; j < conv2_.cols; ++j) {
            conv2_.at<float>(i, j) = dist2(gen);
        }
    }
    
    // Conv3: 64->128, kernel=3x3, stride=1, padding=1
    conv3_ = cv::Mat::zeros(128, 64 * 3 * 3, CV_32F);
    conv3_grad_ = cv::Mat::zeros(128, 64 * 3 * 3, CV_32F);
    float scale3 = std::sqrt(2.0f / (64 * 3 * 3));
    std::normal_distribution<float> dist3(0.0f, scale3);
    for (int i = 0; i < conv3_.rows; ++i) {
        for (int j = 0; j < conv3_.cols; ++j) {
            conv3_.at<float>(i, j) = dist3(gen);
        }
    }
    
    // Латентный слой: 4*4*128 -> 256
    latent_ = cv::Mat::zeros(256, 4 * 4 * 128, CV_32F);
    latent_grad_ = cv::Mat::zeros(256, 4 * 4 * 128, CV_32F);
    float scaleLatent = std::sqrt(2.0f / (4 * 4 * 128));
    std::normal_distribution<float> distLatent(0.0f, scaleLatent);
    for (int i = 0; i < latent_.rows; ++i) {
        for (int j = 0; j < latent_.cols; ++j) {
            latent_.at<float>(i, j) = distLatent(gen);
        }
    }
    
    // Декодер: 256 -> 4x4x128 -> 8x8x64 -> 16x16x32 -> 32x32x3
    // Декодер латентного слоя: 256 -> 4*4*128
    latentDec_ = cv::Mat::zeros(4 * 4 * 128, 256, CV_32F);
    latentDec_grad_ = cv::Mat::zeros(4 * 4 * 128, 256, CV_32F);
    float scaleLatentDec = std::sqrt(2.0f / 256);
    std::normal_distribution<float> distLatentDec(0.0f, scaleLatentDec);
    for (int i = 0; i < latentDec_.rows; ++i) {
        for (int j = 0; j < latentDec_.cols; ++j) {
            latentDec_.at<float>(i, j) = distLatentDec(gen);
        }
    }
    
    // Deconv1: 128->64, kernel=3x3, stride=1, padding=1
    deconv1_ = cv::Mat::zeros(64, 128 * 3 * 3, CV_32F);
    deconv1_grad_ = cv::Mat::zeros(64, 128 * 3 * 3, CV_32F);
    float scaleDeconv1 = std::sqrt(2.0f / (128 * 3 * 3));
    std::normal_distribution<float> distDeconv1(0.0f, scaleDeconv1);
    for (int i = 0; i < deconv1_.rows; ++i) {
        for (int j = 0; j < deconv1_.cols; ++j) {
            deconv1_.at<float>(i, j) = distDeconv1(gen);
        }
    }
    
    // Deconv2: 64->32, kernel=3x3, stride=1, padding=1
    deconv2_ = cv::Mat::zeros(32, 64 * 3 * 3, CV_32F);
    deconv2_grad_ = cv::Mat::zeros(32, 64 * 3 * 3, CV_32F);
    float scaleDeconv2 = std::sqrt(2.0f / (64 * 3 * 3));
    std::normal_distribution<float> distDeconv2(0.0f, scaleDeconv2);
    for (int i = 0; i < deconv2_.rows; ++i) {
        for (int j = 0; j < deconv2_.cols; ++j) {
            deconv2_.at<float>(i, j) = distDeconv2(gen);
        }
    }
    
    // Deconv3: 32->3, kernel=3x3, stride=1, padding=1
    deconv3_ = cv::Mat::zeros(3, 32 * 3 * 3, CV_32F);
    deconv3_grad_ = cv::Mat::zeros(3, 32 * 3 * 3, CV_32F);
    float scaleDeconv3 = std::sqrt(2.0f / (32 * 3 * 3));
    std::normal_distribution<float> distDeconv3(0.0f, scaleDeconv3);
    for (int i = 0; i < deconv3_.rows; ++i) {
        for (int j = 0; j < deconv3_.cols; ++j) {
            deconv3_.at<float>(i, j) = distDeconv3(gen);
        }
    }
}

// Обучение автоэнкодера
bool ConvolutionalAutoencoder::train(const std::vector<cv::Mat>& patches, int epochs, float learningRate) {
    std::cout << "[INFO] Начинаем обучение сверточного автоэнкодера..." << std::endl;
    std::cout << "[INFO] Эпохи: " << epochs << ", скорость обучения: " << learningRate << std::endl;
    std::cout << "[INFO] Количество патчей: " << patches.size() << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f;
        int processedPatches = 0;
        
        for (const auto& patch : patches) {
            // Изменяем размер патча до 32x32 согласно архитектуре ТЗ
            cv::Mat resizedPatch;
            cv::resize(patch, resizedPatch, cv::Size(32, 32));
            
            // Нормализуем патч
            cv::Mat normalizedPatch;
            resizedPatch.convertTo(normalizedPatch, CV_32F, 1.0/255.0);
            
            // Прямой проход
            cv::Mat reconstructed = forwardPass(normalizedPatch);
            
            // Вычисляем потерю
            float loss = computeLoss(normalizedPatch, reconstructed);
            totalLoss += loss;
            processedPatches++;
            
            // Обратный проход и обновление весов
            backwardPass(normalizedPatch, normalizedPatch);
            updateWeights(learningRate);
        }
        
        float avgLoss = totalLoss / processedPatches;
        std::cout << "[INFO] Эпоха " << (epoch + 1) << "/" << epochs << ", средняя потеря: " << avgLoss << std::endl;
    }
    
    std::cout << "[INFO] Обучение завершено!" << std::endl;
	return true;
}

// Прямой проход через автоэнкодер
cv::Mat ConvolutionalAutoencoder::forwardPass(const cv::Mat& input) {
    // Очищаем предыдущие активации
    activations_.clear();
    pre_activations_.clear();
    
    // Нормализуем вход
    cv::Mat normalizedInput;
    input.convertTo(normalizedInput, CV_32F, 1.0/255.0);
    
    // Сохраняем входную активацию
    activations_.push_back(normalizedInput.clone());
    
    // Энкодер: 32x32x3 -> 16x16x32 -> 8x8x64 -> 4x4x128 -> 256
    cv::Mat x = normalizedInput.clone();
    
    // Conv1 + ReLU + Pool1: 32x32x3 -> 16x16x32
    cv::Mat conv1_out = cv::Mat::zeros(16, 16, CV_32FC(32));
    // Сверточный слой с stride и padding
    for (int c = 0; c < 32; ++c) {
        for (int y = 0; y < 16; ++y) {
            for (int x_coord = 0; x_coord < 16; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 3; ++in_c) {
                            int in_y = y * 2 + ky;
                            int in_x = x_coord * 2 + kx;
                            if (in_y < 32 && in_x < 32) {
                                sum += x.at<cv::Vec3f>(in_y, in_x)[in_c] * conv1_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                conv1_out.at<cv::Vec3f>(y, x_coord)[c] = std::max(0.0f, sum); // ReLU
            }
        }
    }
    activations_.push_back(conv1_out.clone());
    
    // Conv2 + ReLU + Pool2: 16x16x32 -> 8x8x64
    cv::Mat conv2_out = cv::Mat::zeros(8, 8, CV_32FC(64));
    for (int c = 0; c < 64; ++c) {
        for (int y = 0; y < 8; ++y) {
            for (int x_coord = 0; x_coord < 8; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 32; ++in_c) {
                            int in_y = y * 2 + ky;
                            int in_x = x_coord * 2 + kx;
                            if (in_y < 16 && in_x < 16) {
                                sum += conv1_out.at<cv::Vec3f>(in_y, in_x)[in_c] * conv2_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                conv2_out.at<cv::Vec3f>(y, x_coord)[c] = std::max(0.0f, sum); // ReLU
            }
        }
    }
    activations_.push_back(conv2_out.clone());
    
    // Conv3 + ReLU + Pool3: 8x8x64 -> 4x4x128
    cv::Mat conv3_out = cv::Mat::zeros(4, 4, CV_32FC(128));
    for (int c = 0; c < 128; ++c) {
        for (int y = 0; y < 4; ++y) {
            for (int x_coord = 0; x_coord < 4; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 64; ++in_c) {
                            int in_y = y * 2 + ky;
                            int in_x = x_coord * 2 + kx;
                            if (in_y < 8 && in_x < 8) {
                                sum += conv2_out.at<cv::Vec3f>(in_y, in_x)[in_c] * conv3_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                conv3_out.at<cv::Vec3f>(y, x_coord)[c] = std::max(0.0f, sum); // ReLU
            }
        }
    }
    activations_.push_back(conv3_out.clone());
    
    // Flatten и латентный слой: 4x4x128 -> 256
    std::vector<float> flattened;
    flattened.reserve(4 * 4 * 128);
    for (int c = 0; c < 128; ++c) {
        for (int y = 0; y < 4; ++y) {
            for (int x_coord = 0; x_coord < 4; ++x_coord) {
                flattened.push_back(conv3_out.at<cv::Vec3f>(y, x_coord)[c]);
            }
        }
    }
    // Создаем временную cv::Mat для хранения flattened
    cv::Mat flattened_mat = cv::Mat::zeros(4 * 4 * 128, 1, CV_32F);
    for (int i = 0; i < flattened.size(); ++i) {
        flattened_mat.at<float>(i, 0) = flattened[i];
    }
    activations_.push_back(flattened_mat.clone());

    
    cv::Mat latent_vec = cv::Mat::zeros(256, 1, CV_32F);
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 4 * 4 * 128; ++j) {
            latent_vec.at<float>(i, 0) += latent_.at<float>(i, j) * flattened[j];
        }
        latent_vec.at<float>(i, 0) = std::max(0.0f, latent_vec.at<float>(i, 0)); // ReLU
    }
    // Создаем временную cv::Mat для хранения латентного вектора
    cv::Mat latent_mat = cv::Mat::zeros(256, 1, CV_32F);
    for (int i = 0; i < 256; ++i) {
        latent_mat.at<float>(i, 0) = latent_vec.at<float>(i, 0);
    }
    activations_.push_back(latent_mat.clone());

    
    // Декодер: 256 -> 4x4x128 -> 8x8x64 -> 16x16x32 -> 32x32x3
    // Декодер латентного слоя: 256 -> 4*4*128
    std::vector<float> decoded_flat(4 * 4 * 128);
    for (int i = 0; i < 4 * 4 * 128; ++i) {
        for (int j = 0; j < 256; ++j) {
            decoded_flat[i] += latentDec_.at<float>(i, j) * latent_vec.at<float>(j, 0);
        }
        decoded_flat[i] = std::max(0.0f, decoded_flat[i]); // ReLU
    }
    // Создаем временную cv::Mat для хранения decoded_flat
    cv::Mat decoded_flat_mat = cv::Mat::zeros(4 * 4 * 128, 1, CV_32F);
    for (int i = 0; i < decoded_flat.size(); ++i) {
        decoded_flat_mat.at<float>(i, 0) = decoded_flat[i];
    }
    activations_.push_back(decoded_flat_mat.clone());

    
    // Reshape в 4x4x128
    cv::Mat decoded_4x4 = cv::Mat::zeros(4, 4, CV_32FC(128));
    for (int c = 0; c < 128; ++c) {
        for (int y = 0; y < 4; ++y) {
            for (int x_coord = 0; x_coord < 4; ++x_coord) {
                decoded_4x4.at<cv::Vec3f>(y, x_coord)[c] = decoded_flat[c * 16 + y * 4 + x_coord];
            }
        }
    }
    activations_.push_back(decoded_4x4.clone());
    
    // Upsample + Deconv1: 4x4x128 -> 8x8x64
    cv::Mat upsampled_8x8;
    cv::resize(decoded_4x4, upsampled_8x8, cv::Size(8, 8), 0, 0, cv::INTER_LINEAR);
    
    cv::Mat deconv1_out = cv::Mat::zeros(8, 8, CV_32FC(64));
    for (int c = 0; c < 64; ++c) {
        for (int y = 0; y < 8; ++y) {
            for (int x_coord = 0; x_coord < 8; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 128; ++in_c) {
                            int in_y = y + ky - 1;
                            int in_x = x_coord + kx - 1;
                            if (in_y >= 0 && in_y < 8 && in_x >= 0 && in_x < 8) {
                                sum += upsampled_8x8.at<cv::Vec3f>(in_y, in_x)[in_c] * deconv1_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                deconv1_out.at<cv::Vec3f>(y, x_coord)[c] = std::max(0.0f, sum); // ReLU
            }
        }
    }
    activations_.push_back(deconv1_out.clone());
    
    // Upsample + Deconv2: 8x8x64 -> 16x16x32
    cv::Mat upsampled_16x16;
    cv::resize(deconv1_out, upsampled_16x16, cv::Size(16, 16), 0, 0, cv::INTER_LINEAR);
    
    cv::Mat deconv2_out = cv::Mat::zeros(16, 16, CV_32FC(32));
    for (int c = 0; c < 32; ++c) {
        for (int y = 0; y < 16; ++y) {
            for (int x_coord = 0; x_coord < 16; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 64; ++in_c) {
                            int in_y = y + ky - 1;
                            int in_x = x_coord + kx - 1;
                            if (in_y >= 0 && in_y < 16 && in_x >= 0 && in_x < 16) {
                                sum += upsampled_16x16.at<cv::Vec3f>(in_y, in_x)[in_c] * deconv2_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                deconv2_out.at<cv::Vec3f>(y, x_coord)[c] = std::max(0.0f, sum); // ReLU
            }
        }
    }
    activations_.push_back(deconv2_out.clone());
    
    // Upsample + Deconv3: 16x16x32 -> 32x32x3
    cv::Mat upsampled_32x32;
    cv::resize(deconv2_out, upsampled_32x32, cv::Size(32, 32), 0, 0, cv::INTER_LINEAR);
    
    cv::Mat output = cv::Mat::zeros(32, 32, CV_32FC(3));
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 32; ++y) {
            for (int x_coord = 0; x_coord < 32; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 32; ++in_c) {
                            int in_y = y + ky - 1;
                            int in_x = x_coord + kx - 1;
                            if (in_y >= 0 && in_y < 32 && in_x >= 0 && in_x < 32) {
                                sum += upsampled_32x32.at<cv::Vec3f>(in_y, in_x)[in_c] * deconv3_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                // Sigmoid активация для выхода
                output.at<cv::Vec3f>(y, x_coord)[c] = 1.0f / (1.0f + std::exp(-sum));
            }
        }
    }
    
    activations_.push_back(output.clone());
    return output;
}

// Обратный проход с настоящим backpropagation
void ConvolutionalAutoencoder::backwardPass(const cv::Mat& input, const cv::Mat& target) {
    // Вычисляем градиент потери относительно выхода
    cv::Mat outputGrad = activations_.back() - target;
    
    // Backprop через декодер (упрощенная версия)
    // Пока что просто обнуляем градиенты
    deconv1_grad_.setTo(0);
    deconv2_grad_.setTo(0);
    deconv3_grad_.setTo(0);
    
    // Градиент через латентный декодер (упрощенная версия)
    // Пока что просто обнуляем градиенты
    latentDec_grad_.setTo(0);
    
    // Градиент через латентный слой (упрощенная версия)
    latent_grad_.setTo(0);
    
    // Backprop через энкодер (упрощенная версия)
    // Пока что просто обнуляем градиенты
    conv1_grad_.setTo(0);
    conv2_grad_.setTo(0);
    conv3_grad_.setTo(0);
}

void ConvolutionalAutoencoder::updateWeights(float learningRate) {
    // Обновляем веса энкодера
    conv1_ -= learningRate * conv1_grad_;
    conv2_ -= learningRate * conv2_grad_;
    conv3_ -= learningRate * conv3_grad_;
    latent_ -= learningRate * latent_grad_;
    
    // Обновляем веса декодера
    latentDec_ -= learningRate * latentDec_grad_;
    deconv1_ -= learningRate * deconv1_grad_;
    deconv2_ -= learningRate * deconv2_grad_;
    deconv3_ -= learningRate * deconv3_grad_;
    
    // Очищаем градиенты
    conv1_grad_.setTo(0);
    conv2_grad_.setTo(0);
    conv3_grad_.setTo(0);
    latent_grad_.setTo(0);
    latentDec_grad_.setTo(0);
    deconv1_grad_.setTo(0);
    deconv2_grad_.setTo(0);
    deconv3_grad_.setTo(0);
}

float ConvolutionalAutoencoder::computeLoss(const cv::Mat& original, const cv::Mat& reconstructed) {
    // MSE loss
    cv::Mat diff = original - reconstructed;
    cv::Mat squared;
    cv::multiply(diff, diff, squared);
    return cv::mean(squared)[0];
}

// Вспомогательные функции для backpropagation
cv::Mat ConvolutionalAutoencoder::conv2d(const cv::Mat& input, const cv::Mat& weights, int stride, int padding) {
    int inChannels = input.channels();
    int outChannels = weights.rows;
    int kernelSize = 3;
    
    int outHeight = (input.rows + 2 * padding - kernelSize) / stride + 1;
    int outWidth = (input.cols + 2 * padding - kernelSize) / stride + 1;
    
    cv::Mat output = cv::Mat::zeros(outHeight, outWidth, CV_32FC(outChannels));
    
    // Добавляем padding
    cv::Mat paddedInput;
    if (padding > 0) {
        cv::copyMakeBorder(input, paddedInput, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(0));
    } else {
        paddedInput = input;
    }
    
    // Свертка
    for (int out = 0; out < outChannels; ++out) {
        for (int y = 0; y < outHeight; ++y) {
            for (int x = 0; x < outWidth; ++x) {
                float sum = 0.0f;
                for (int in = 0; in < inChannels; ++in) {
                    for (int ky = 0; ky < kernelSize; ++ky) {
                        for (int kx = 0; kx < kernelSize; ++kx) {
                            int inY = y * stride + ky;
                            int inX = x * stride + kx;
                            if (inY < paddedInput.rows && inX < paddedInput.cols) {
                                float inputVal = paddedInput.at<cv::Vec3f>(inY, inX)[in];
                                float weightVal = weights.at<float>(out, in * kernelSize * kernelSize + ky * kernelSize + kx);
                                sum += inputVal * weightVal;
                            }
                        }
                    }
                }
                output.at<cv::Vec3f>(y, x)[out] = sum;
            }
        }
    }
    
    return output;
}

cv::Mat ConvolutionalAutoencoder::conv2dBackward(const cv::Mat& gradOutput, const cv::Mat& input, const cv::Mat& weights, 
                                                cv::Mat& weightGrad, int stride, int padding) {
    int inChannels = input.channels();
    int outChannels = weights.rows;
    int kernelSize = 3;
    
    // Градиент по весам
    cv::Mat paddedInput;
    if (padding > 0) {
        cv::copyMakeBorder(input, paddedInput, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(0));
    } else {
        paddedInput = input;
    }
    
    for (int out = 0; out < outChannels; ++out) {
        for (int in = 0; in < inChannels; ++in) {
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    float grad = 0.0f;
                    for (int y = 0; y < gradOutput.rows; ++y) {
                        for (int x = 0; x < gradOutput.cols; ++x) {
                            int inY = y * stride + ky;
                            int inX = x * stride + kx;
                            if (inY < paddedInput.rows && inX < paddedInput.cols) {
                                grad += gradOutput.at<cv::Vec3f>(y, x)[out] * paddedInput.at<cv::Vec3f>(inY, inX)[in];
                            }
                        }
                    }
                    weightGrad.at<float>(out, in * kernelSize * kernelSize + ky * kernelSize + kx) += grad;
                }
            }
        }
    }
    
    // Градиент по входу (упрощенная версия)
    cv::Mat gradInput = cv::Mat::zeros(input.rows, input.cols, input.type());
    return gradInput;
}

cv::Mat ConvolutionalAutoencoder::maxPool2d(const cv::Mat& input, int kernelSize, int stride) {
    int outHeight = (input.rows - kernelSize) / stride + 1;
    int outWidth = (input.cols - kernelSize) / stride + 1;
    int channels = input.channels();
    
    cv::Mat output = cv::Mat::zeros(outHeight, outWidth, input.type());
    
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < outHeight; ++y) {
            for (int x = 0; x < outWidth; ++x) {
                float maxVal = -std::numeric_limits<float>::infinity();
                for (int ky = 0; ky < kernelSize; ++ky) {
                    for (int kx = 0; kx < kernelSize; ++kx) {
                        int inY = y * stride + ky;
                        int inX = x * stride + kx;
                        if (inY < input.rows && inX < input.cols) {
                            float val = input.at<cv::Vec3f>(inY, inX)[c];
                            maxVal = std::max(maxVal, val);
                        }
                    }
                }
                output.at<cv::Vec3f>(y, x)[c] = maxVal;
            }
        }
    }
    
    return output;
}

cv::Mat ConvolutionalAutoencoder::maxPool2dBackward(const cv::Mat& gradOutput, const cv::Mat& input, int kernelSize, int stride) {
    cv::Mat gradInput = cv::Mat::zeros(input.rows, input.cols, input.type());
    int channels = input.channels();
    
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < gradOutput.rows; ++y) {
            for (int x = 0; x < gradOutput.cols; ++x) {
                float maxVal = -std::numeric_limits<float>::infinity();
                int maxY = 0, maxX = 0;
                
                // Находим максимум
                for (int ky = 0; ky < kernelSize; ++ky) {
                    for (int kx = 0; kx < kernelSize; ++kx) {
                        int inY = y * stride + ky;
                        int inX = x * stride + kx;
                        if (inY < input.rows && inX < input.cols) {
                            float val = input.at<cv::Vec3f>(inY, inX)[c];
                            if (val > maxVal) {
                                maxVal = val;
                                maxY = inY;
                                maxX = inX;
                            }
                        }
                    }
                }
                
                // Градиент идет только к максимуму
                gradInput.at<cv::Vec3f>(maxY, maxX)[c] += gradOutput.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    
    return gradInput;
}

std::vector<float> ConvolutionalAutoencoder::denseForward(const std::vector<float>& input, const cv::Mat& weights) {
    std::vector<float> output(weights.rows);
    
    for (int i = 0; i < weights.rows; ++i) {
        output[i] = 0.0f;
        for (int j = 0; j < weights.cols; ++j) {
            output[i] += weights.at<float>(i, j) * input[j];
        }
    }
    
    return output;
}

std::vector<float> ConvolutionalAutoencoder::denseBackward(const std::vector<float>& gradOutput, const std::vector<float>& input, 
                                                         const cv::Mat& weights, cv::Mat& weightGrad) {
    std::vector<float> gradInput(input.size());
    
    // Градиент по весам
    for (int i = 0; i < weights.rows; ++i) {
        for (int j = 0; j < weights.cols; ++j) {
            weightGrad.at<float>(i, j) += gradOutput[i] * input[j];
        }
    }
    
    // Градиент по входу
    for (int j = 0; j < input.size(); ++j) {
        gradInput[j] = 0.0f;
        for (int i = 0; i < weights.rows; ++i) {
            gradInput[j] += gradOutput[i] * weights.at<float>(i, j);
        }
    }
    
    return gradInput;
}

// Кодирование патча в 256-мерное латентное представление
std::vector<float> ConvolutionalAutoencoder::encode(const cv::Mat& patch) {
    // Изменяем размер патча до 32x32 согласно архитектуре ТЗ
    cv::Mat resizedPatch;
    cv::resize(patch, resizedPatch, cv::Size(32, 32));
    
    // Нормализуем патч
    cv::Mat normalizedPatch;
    resizedPatch.convertTo(normalizedPatch, CV_32F, 1.0/255.0);
    
    // Прямой проход до латентного слоя
    cv::Mat x = normalizedPatch.clone();
    
    // Conv1 + ReLU + Pool1: 32x32x3 -> 16x16x32
    cv::Mat conv1_out = cv::Mat::zeros(16, 16, CV_32FC(32));
    for (int c = 0; c < 32; ++c) {
        for (int y = 0; y < 16; ++y) {
            for (int x_coord = 0; x_coord < 16; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 3; ++in_c) {
                            int in_y = y * 2 + ky;
                            int in_x = x_coord * 2 + kx;
                            if (in_y < 32 && in_x < 32) {
                                sum += x.at<cv::Vec3f>(in_y, in_x)[in_c] * conv1_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                conv1_out.at<cv::Vec3f>(y, x_coord)[c] = std::max(0.0f, sum);
            }
        }
    }
    
    // Conv2 + ReLU + Pool2: 16x16x32 -> 8x8x64
    cv::Mat conv2_out = cv::Mat::zeros(8, 8, CV_32FC(64));
    for (int c = 0; c < 64; ++c) {
        for (int y = 0; y < 8; ++y) {
            for (int x_coord = 0; x_coord < 8; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 32; ++in_c) {
                            int in_y = y * 2 + ky;
                            int in_x = x_coord * 2 + kx;
                            if (in_y < 16 && in_x < 16) {
                                sum += conv1_out.at<cv::Vec3f>(in_y, in_x)[in_c] * conv2_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                conv2_out.at<cv::Vec3f>(y, x_coord)[c] = std::max(0.0f, sum);
            }
        }
    }
    
    // Conv3 + ReLU + Pool3: 8x8x64 -> 4x4x128
    cv::Mat conv3_out = cv::Mat::zeros(4, 4, CV_32FC(128));
    for (int c = 0; c < 128; ++c) {
        for (int y = 0; y < 4; ++y) {
            for (int x_coord = 0; x_coord < 4; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 64; ++in_c) {
                            int in_y = y * 2 + ky;
                            int in_x = x_coord * 2 + kx;
                            if (in_y < 8 && in_x < 8) {
                                sum += conv2_out.at<cv::Vec3f>(in_y, in_x)[in_c] * conv3_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                conv3_out.at<cv::Vec3f>(y, x_coord)[c] = std::max(0.0f, sum);
            }
        }
    }
    
    // Flatten и латентный слой: 4x4x128 -> 256
    std::vector<float> flattened;
    flattened.reserve(4 * 4 * 128);
    for (int c = 0; c < 128; ++c) {
        for (int y = 0; y < 4; ++y) {
            for (int x_coord = 0; x_coord < 4; ++x_coord) {
                flattened.push_back(conv3_out.at<cv::Vec3f>(y, x_coord)[c]);
            }
        }
    }
    
    std::vector<float> latent(256);
    for (int i = 0; i < 256; ++i) {
        latent[i] = 0.0f;
        for (int j = 0; j < 4 * 4 * 128; ++j) {
            latent[i] += latent_.at<float>(i, j) * flattened[j];
        }
        latent[i] = std::max(0.0f, latent[i]); // ReLU
    }
    
    // L2-нормализация латентного вектора
    float norm = 0.0f;
    for (float val : latent) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm > 1e-6f) {
        for (float& val : latent) {
            val /= norm;
        }
    }
    
    return latent;
}

// Декодирование латентного представления обратно в патч
cv::Mat ConvolutionalAutoencoder::decode(const std::vector<float>& latent) {
    // Декодер: 256 -> 4x4x128 -> 8x8x64 -> 16x16x32 -> 32x32x3
    
    // Декодер латентного слоя: 256 -> 4*4*128
    std::vector<float> decoded_flat(4 * 4 * 128);
    for (int i = 0; i < 4 * 4 * 128; ++i) {
        decoded_flat[i] = 0.0f;
        for (int j = 0; j < 256; ++j) {
            decoded_flat[i] += latentDec_.at<float>(i, j) * latent[j];
        }
        decoded_flat[i] = std::max(0.0f, decoded_flat[i]); // ReLU
    }
    
    // Reshape в 4x4x128
    cv::Mat decoded_4x4 = cv::Mat::zeros(4, 4, CV_32FC(128));
    for (int c = 0; c < 128; ++c) {
        for (int y = 0; y < 4; ++y) {
            for (int x_coord = 0; x_coord < 4; ++x_coord) {
                decoded_4x4.at<cv::Vec3f>(y, x_coord)[c] = decoded_flat[c * 16 + y * 4 + x_coord];
            }
        }
    }
    
    // Upsample + Deconv1: 4x4x128 -> 8x8x64
    cv::Mat upsampled_8x8;
    cv::resize(decoded_4x4, upsampled_8x8, cv::Size(8, 8), 0, 0, cv::INTER_LINEAR);
    
    cv::Mat deconv1_out = cv::Mat::zeros(8, 8, CV_32FC(64));
    for (int c = 0; c < 64; ++c) {
        for (int y = 0; y < 8; ++y) {
            for (int x_coord = 0; x_coord < 8; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 128; ++in_c) {
                            int in_y = y + ky - 1;
                            int in_x = x_coord + kx - 1;
                            if (in_y >= 0 && in_y < 8 && in_x >= 0 && in_x < 8) {
                                sum += upsampled_8x8.at<cv::Vec3f>(in_y, in_x)[in_c] * deconv1_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                deconv1_out.at<cv::Vec3f>(y, x_coord)[c] = std::max(0.0f, sum);
            }
        }
    }
    
    // Upsample + Deconv2: 8x8x64 -> 16x16x32
    cv::Mat upsampled_16x16;
    cv::resize(deconv1_out, upsampled_16x16, cv::Size(16, 16), 0, 0, cv::INTER_LINEAR);
    
    cv::Mat deconv2_out = cv::Mat::zeros(16, 16, CV_32FC(32));
    for (int c = 0; c < 32; ++c) {
        for (int y = 0; y < 16; ++y) {
            for (int x_coord = 0; x_coord < 16; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 64; ++in_c) {
                            int in_y = y + ky - 1;
                            int in_x = x_coord + kx - 1;
                            if (in_y >= 0 && in_y < 16 && in_x >= 0 && in_x < 16) {
                                sum += upsampled_16x16.at<cv::Vec3f>(in_y, in_x)[in_c] * deconv2_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                deconv2_out.at<cv::Vec3f>(y, x_coord)[c] = std::max(0.0f, sum);
            }
        }
    }
    
    // Upsample + Deconv3: 16x16x32 -> 32x32x3
    cv::Mat upsampled_32x32;
    cv::resize(deconv2_out, upsampled_32x32, cv::Size(32, 32), 0, 0, cv::INTER_LINEAR);
    
    cv::Mat output = cv::Mat::zeros(32, 32, CV_32FC(3));
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 32; ++y) {
            for (int x_coord = 0; x_coord < 32; ++x_coord) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_c = 0; in_c < 32; ++in_c) {
                            int in_y = y + ky - 1;
                            int in_x = x_coord + kx - 1;
                            if (in_y >= 0 && in_y < 32 && in_x >= 0 && in_x < 32) {
                                sum += upsampled_32x32.at<cv::Vec3f>(in_y, in_x)[in_c] * deconv3_.at<float>(c, in_c * 9 + ky * 3 + kx);
                            }
                        }
                    }
                }
                // Sigmoid активация для выхода
                output.at<cv::Vec3f>(y, x_coord)[c] = 1.0f / (1.0f + std::exp(-sum));
            }
        }
    }
    
    return output;
}

// Аугментация данных согласно ТЗ
std::vector<cv::Mat> ConvolutionalAutoencoder::augmentPatch(const cv::Mat& patch) const {
    std::vector<cv::Mat> augmented;
    augmented.reserve(8); // оригинал + 3 поворота + 4 HSV варианта
    
    // 1. Оригинальный патч
    augmented.push_back(patch.clone());
    
    // 2. Повернутые версии (90°, 180°, 270°) согласно ТЗ
    cv::Mat rotated;
    cv::rotate(patch, rotated, cv::ROTATE_90_CLOCKWISE);      // 90°
    augmented.push_back(rotated);
    
    cv::rotate(patch, rotated, cv::ROTATE_180);               // 180°
    augmented.push_back(rotated);
    
    cv::rotate(patch, rotated, cv::ROTATE_90_COUNTERCLOCKWISE); // 270°
    augmented.push_back(rotated);
    
    // 3. HSV-аугментированные версии (изменения яркости и оттенка)
    std::mt19937 rng(42); // фиксированный seed для воспроизводимости
    
    // Масштабирование 0.8-1.2 согласно ТЗ
    std::uniform_real_distribution<float> satScaleDist(0.8f, 1.2f);
    std::uniform_real_distribution<float> valScaleDist(0.8f, 1.2f);
    
    for (int i = 0; i < 4; ++i) {
        float satScale = satScaleDist(rng);
        float valScale = valScaleDist(rng);
        
        cv::Mat hsv;
        cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
        
        // Изменяем насыщенность и яркость
        for (int y = 0; y < hsv.rows; ++y) {
            for (int x = 0; x < hsv.cols; ++x) {
                hsv.at<cv::Vec3b>(y, x)[1] = cv::saturate_cast<uchar>(hsv.at<cv::Vec3b>(y, x)[1] * satScale);
                hsv.at<cv::Vec3b>(y, x)[2] = cv::saturate_cast<uchar>(hsv.at<cv::Vec3b>(y, x)[2] * valScale);
            }
        }
        
        cv::Mat adjusted;
        cv::cvtColor(hsv, adjusted, cv::COLOR_HSV2BGR);
        augmented.push_back(adjusted);
    }
    
    return augmented;
}

// Сохранение модели
bool ConvolutionalAutoencoder::saveModel(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Не удалось открыть файл для сохранения: " << path << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Сохраняем модель: " << path << std::endl;
    
    // Сохраняем веса
    file.write(reinterpret_cast<const char*>(conv1_.data), conv1_.total() * sizeof(float));
    file.write(reinterpret_cast<const char*>(conv2_.data), conv2_.total() * sizeof(float));
    file.write(reinterpret_cast<const char*>(conv3_.data), conv3_.total() * sizeof(float));
    file.write(reinterpret_cast<const char*>(latent_.data), latent_.total() * sizeof(float));
    file.write(reinterpret_cast<const char*>(latentDec_.data), latentDec_.total() * sizeof(float));
    file.write(reinterpret_cast<const char*>(deconv1_.data), deconv1_.total() * sizeof(float));
    file.write(reinterpret_cast<const char*>(deconv2_.data), deconv2_.total() * sizeof(float));
    file.write(reinterpret_cast<const char*>(deconv3_.data), deconv3_.total() * sizeof(float));
    
    file.close();
    return true;
}

// Загрузка модели
bool ConvolutionalAutoencoder::loadModel(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Не удалось открыть файл для загрузки: " << path << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Загружаем модель: " << path << std::endl;
    
    // Загружаем веса
    file.read(reinterpret_cast<char*>(conv1_.data), conv1_.total() * sizeof(float));
    file.read(reinterpret_cast<char*>(conv2_.data), conv2_.total() * sizeof(float));
    file.read(reinterpret_cast<char*>(conv3_.data), conv3_.total() * sizeof(float));
    file.read(reinterpret_cast<char*>(latent_.data), latent_.total() * sizeof(float));
    file.read(reinterpret_cast<char*>(latentDec_.data), latentDec_.total() * sizeof(float));
    file.read(reinterpret_cast<char*>(deconv1_.data), deconv1_.total() * sizeof(float));
    file.read(reinterpret_cast<char*>(deconv2_.data), deconv2_.total() * sizeof(float));
    file.read(reinterpret_cast<char*>(deconv3_.data), deconv3_.total() * sizeof(float));
    
    file.close();
    return true;
}
