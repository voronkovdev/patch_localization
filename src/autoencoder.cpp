#include "autoencoder.h"
#include "image_processor.h"
#include <random>

#ifdef USE_TINYDNN
#include <tiny_dnn/tiny_dnn.h>
namespace tdnn = tiny_dnn;
#endif

// Извлекает 256-мерный дескриптор из патча
// Использует простую HSV-гистограмму: 8x8x4 = 256 измерений
std::vector<float> Autoencoder::encode(const cv::Mat& patchBgr) const {
	CV_Assert(!patchBgr.empty());
	
	// Конвертируем BGR в HSV для лучшего анализа цвета
	cv::Mat hsv;
	cv::cvtColor(patchBgr, hsv, cv::COLOR_BGR2HSV);
	
	// Размеры гистограммы: 8x8x4 = 256 измерений
	int histSizeH = 8;   // Оттенок (Hue) - 8 бинов
	int histSizeS = 8;   // Насыщенность (Saturation) - 8 бинов  
	int histSizeV = 4;   // Яркость (Value) - 4 бина
	
	// Диапазоны значений HSV
	float rangeH[] = {0, 180};  // Оттенок: 0-179 (OpenCV использует 0-179)
	float rangeS[] = {0, 256};  // Насыщенность: 0-255
	float rangeV[] = {0, 256};  // Яркость: 0-255
	
	const float* ranges[] = { rangeH, rangeS, rangeV };
	int channels[] = {0, 1, 2};  // H, S, V каналы
	int histSize[] = {histSizeH, histSizeS, histSizeV};
	
	// Вычисляем 3D гистограмму
	cv::Mat hist;
	cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 3, histSize, ranges, true, false);
	
	// Преобразуем в одномерный вектор
	hist = hist.reshape(1, 1);
	
	// Нормализуем к единичной сумме для инвариантности к яркости
	if (cv::sum(hist)[0] > 0) {
		cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
	}
	
	// Копируем в std::vector
	std::vector<float> vec(hist.cols);
	for (int i = 0; i < hist.cols; ++i) {
		vec[i] = static_cast<float>(hist.at<float>(0, i));
	}
	
	// Убеждаемся, что размер точно 256
	if (vec.size() < 256) {
		vec.resize(256, 0.0f);  // дополняем нулями
	}
	if (vec.size() > 256) {
		vec.resize(256);  // обрезаем
	}
	
	return vec;
}

// Создает аугментированные версии патча согласно ТЗ
std::vector<cv::Mat> Autoencoder::augmentPatch(const cv::Mat& patch) const {
	std::vector<cv::Mat> augmented;
	augmented.reserve(8); // оригинал + 4 поворота + 3 HSV варианта
	
	// 1. Оригинальный патч
	augmented.push_back(patch.clone());
	
	// 2. Повернутые версии (0°, 90°, 180°, 270°) согласно ТЗ
	for (int rot = 1; rot < 4; ++rot) {
		cv::Mat rotated;
		if (rot == 1) {
			cv::rotate(patch, rotated, cv::ROTATE_90_CLOCKWISE);      // 90°
		} else if (rot == 2) {
			cv::rotate(patch, rotated, cv::ROTATE_180);               // 180°
		} else if (rot == 3) {
			cv::rotate(patch, rotated, cv::ROTATE_90_COUNTERCLOCKWISE); // 270°
		}
		augmented.push_back(rotated);
	}
	
	// 3. HSV-аугментированные версии (изменения яркости и оттенка)
	std::mt19937 rng(42); // фиксированный seed для воспроизводимости

	// Масштабирование 0.8-1.2 согласно ТЗ
	std::uniform_real_distribution<float> satScaleDist(0.8f, 1.2f);
	std::uniform_real_distribution<float> valScaleDist(0.8f, 1.2f);

	for (int i = 0; i < 4; ++i) {
		float satScale = satScaleDist(rng);
		float valScale = valScaleDist(rng);
		
		cv::Mat adjusted = ImageProcessor::adjustHSV(patch, 0.0, satScale, valScale);
		augmented.push_back(adjusted);
	}
	
	return augmented;
}

// Обучает сверточный автоэнкодер согласно ТЗ
bool Autoencoder::train(const std::vector<cv::Mat>& trainPatches, int epochs, float learningRate, const std::string& modelPath) {
#ifdef USE_TINYDNN
	if (trainPatches.empty()) return false;
	
	try {
		// Сверточная сеть согласно ТЗ: 64×64 → 32×32×16 → 16×16×32 → 8×8×64 → 256
		tdnn::network<tdnn::sequential> net;
		
		// Энкодер (сверточные слои)
		net << tdnn::layers::conv(64, 64, 3, 1, 16, tdnn::padding::same, true, 1, 1, 1, 1)  // 64×64×1 → 64×64×16
			<< tdnn::activation::tanh()
			<< tdnn::layers::ave_pool(64, 64, 16, 2)                             // 64×64×16 → 32×32×16
			<< tdnn::layers::conv(32, 32, 3, 16, 32, tdnn::padding::same, true, 1, 1, 1, 1) // 32×32×16 → 32×32×32
			<< tdnn::activation::tanh()
			<< tdnn::layers::ave_pool(32, 32, 32, 2)                             // 32×32×32 → 16×16×32
			<< tdnn::layers::conv(16, 16, 3, 32, 64, tdnn::padding::same, true, 1, 1, 1, 1) // 16×16×32 → 16×16×64
			<< tdnn::activation::tanh()
			<< tdnn::layers::ave_pool(16, 16, 64, 2)                             // 16×16×64 → 8×8×64
			<< tdnn::layers::fc(8*8*64, 256, true)                               // 8×8×64 → 256
			<< tdnn::activation::tanh();
		
		// Декодер (обратные сверточные слои)
		net << tdnn::layers::fc(256, 8*8*64, true)                               // 256 → 8×8×64
			<< tdnn::activation::tanh()
			<< tdnn::layers::deconv(8, 8, 3, 64, 32, tdnn::padding::same, true, 1, 1) // 8×8×64 → 16×16×32
			<< tdnn::activation::tanh()
			<< tdnn::layers::deconv(16, 16, 3, 32, 16, tdnn::padding::same, true, 1, 1) // 16×16×32 → 32×32×16
			<< tdnn::activation::tanh()
			<< tdnn::layers::deconv(32, 32, 3, 16, 1, tdnn::padding::same, true, 1, 1); // 32×32×16 → 64×64×1
		
		// Подготавливаем данные для обучения (сверточная сеть работает с изображениями)
		std::vector<tdnn::vec_t> train_data;
		std::vector<tdnn::vec_t> train_labels;
		
		for (const auto& patch : trainPatches) {
			// Приводим патч к размеру 64×64 согласно ТЗ (для сверточной сети)
			cv::Mat resizedPatch;
			cv::resize(patch, resizedPatch, cv::Size(64, 64));
			
			// Создаем аугментированные версии
			auto augmented = augmentPatch(resizedPatch);
			for (const auto& aug : augmented) {
				// Приводим к размеру 64×64
				cv::Mat resizedAug;
				cv::resize(aug, resizedAug, cv::Size(64, 64));
				
				// Конвертируем в grayscale для упрощения
				cv::Mat gray;
				cv::cvtColor(resizedAug, gray, cv::COLOR_BGR2GRAY);
				
				// Нормализуем к [0,1] для нейронной сети
				tdnn::vec_t input(64*64);
				for (int y = 0; y < 64; ++y) {
					for (int x = 0; x < 64; ++x) {
						input[y*64 + x] = static_cast<float>(gray.at<uchar>(y, x)) / 255.0f;
					}
				}
				
				train_data.push_back(input);
				train_labels.push_back(input); // автоэнкодер: вход = выход
			}
		}
		
		if (train_data.empty()) return false;
		
		// Настраиваем оптимизатор
		tdnn::adam optimizer;
		optimizer.alpha = learningRate;
		
		// Обучаем сеть
		net.fit<tdnn::mse>(optimizer, train_data, train_labels, 
						   static_cast<int>(train_data.size()), epochs);
		
		// Сохраняем модель
		net.save(modelPath);
		m_modelPath = modelPath;
		
		return true;
		
	} catch (const std::exception& e) {
		// В случае ошибки возвращаем false
		return false;
	}
#else
	// Без TinyDNN используем только гистограммы
	return false;
#endif
}

bool Autoencoder::loadModel(const std::string& modelPath) {
#ifdef USE_TINYDNN
	try {
		// Загружаем модель
		tdnn::network<tdnn::sequential> net;
		net.load(modelPath);
		m_modelPath = modelPath;
		return true;
	} catch (const std::exception& e) {
		return false;
	}
#else
	return false;
#endif
}

std::vector<float> Autoencoder::encodeWithModel(const cv::Mat& patchBgr) const {
#ifdef USE_TINYDNN
	if (m_modelPath.empty()) {
		return encode(patchBgr); // fallback к гистограмме
	}
	
	try {
		// Загружаем полную сеть из файла (архитектура + веса)
		tdnn::network<tdnn::sequential> net;
		net.load(m_modelPath);
		
		// Приводим патч к размеру 64×64 согласно ТЗ
		cv::Mat resizedPatch;
		cv::resize(patchBgr, resizedPatch, cv::Size(64, 64));
		
		// Конвертируем в grayscale
		cv::Mat gray;
		cv::cvtColor(resizedPatch, gray, cv::COLOR_BGR2GRAY);
		
		// Нормализуем вход для нейронной сети
		tdnn::vec_t input(64*64);
		for (int y = 0; y < 64; ++y) {
			for (int x = 0; x < 64; ++x) {
				input[y*64 + x] = static_cast<float>(gray.at<uchar>(y, x)) / 255.0f;
			}
		}
		
		// Прямой проход по сети (заполняет буферы выходов слоев)
		(void)net.predict(input);
		
		// Извлекаем 256-мерный выход (узкое место энкодера)
		std::vector<float> latent(256, 0.0f);
		for (auto it = net.begin(); it != net.end(); ++it) {
			auto *layer = *it;
			if (layer && layer->out_data_size() == 256) {
				std::vector<const tdnn::tensor_t*> outs;
				layer->output(outs);
				if (!outs.empty() && outs[0] && !outs[0]->empty()) {
					const tdnn::vec_t &v = (*outs[0])[0];
					for (size_t i = 0; i < 256 && i < v.size(); ++i) {
						latent[i] = static_cast<float>(v[i]);
					}
				}
				break;
			}
		}
		
		return latent;
		
	} catch (const std::exception& e) {
		return encode(patchBgr); // fallback к гистограмме
	}
#else
	return encode(patchBgr);
#endif
} 